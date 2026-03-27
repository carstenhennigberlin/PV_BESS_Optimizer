
import pyomo.environ as pyo
import matplotlib.pyplot as plt

# This module comprises a Pyomo optimization model to schedule the operation of a photovoltaic (PV) system combined 
# with a battery energy storage system (BESS) charged by the PV system only in accordance with the German Innovationsausschreibung.
# The model optimizes the charging and discharging of the BESS to maximize the economic benefit from the PV output and 
# DAA prices, while adhering to various constraints such as power limits, storage capacity, and efficiency.
# In two following steps, the DAA schedule can be sequentially renewed based on IDA and IDC market prices, in order to maximize overall profit.

# Pyomo requires a solver to be installed and specified.
# Ensure you have a solver installed, such as GLPK, CBC, or HiGHS.

# For example HiGHS via pip: 'pip install highspy'

SOLVER = 'highs'  # Specify the solver to use, e.g., 'glpk', 'cbc', 'highs'
PATH_TO_SOLVER = None # Optional: specify the path to the solver executable if needed. for example: "C:\Cbc/bin/cbc.exe"


def get_daa_schedule(pv_output, daa, market_premium, p_limit, storage_capacity, p_charge_max, p_discharge_max, delta_p_perm, number_of_cycles=2, efficiency=0.95, start_soc=0.0, end_soc=0.0):
    """    
    Function to get the DAA schedule for the PV and BESS combination.
    This function initializes the Pyomo model, sets up the variables, constraints, and objective function,
    and then solves the optimization problem.
    
    Parameters:
    ------------
    pv_output (list): List of PV output values for each time step.
    daa (list): List of DAA prices for each time step.
    market_premium (list): list of market premia.
    storage_capacity (float): Maximum storage capacity of the BESS.
    p_charge_max (float): Maximum charging power of the BESS.
    p_discharge_max (float): Maximum discharging power of the BESS.
    delta_p_perm (float): Permissible change in power per time step.
    number_of_cycles (int): Maximum number of cycles for the BESS.
    efficiency (float): Charge and discharge efficiency of the BESS.
    start_soc (float): Initial state of charge of the BESS.
    end_soc (float): End state of charge of the BESS.

    
    Returns:
    -----------
    tuple: A tuple containing the PV output, DAA prices, charge power, discharge power, state of charge, curtailed power, and injection power.
    """
    
    # Get the number of time steps
    num_time_steps = len(daa)

    # Initialize the model
    model = pyo.ConcreteModel()

    # Initialize the time parameters
    model.T = pyo.RangeSet(1, num_time_steps)  # Time step in 1/4 hour
    model.T_plus_1 = pyo.RangeSet(1, num_time_steps + 1)  # Time step in 1/4 hour plus one for state of chargeimport pyomo.environ as pyo

    # Initialize the vectorized variables with model.T elements
    model.p_charge_daa = pyo.Var(model.T, within=pyo.NonNegativeReals) # better within=pyo.NonNegativeReals
    model.p_discharge_daa = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_curtailed_daa = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_inject_daa = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T_plus_1, within=pyo.NonNegativeReals, initialize=0.0)  # State of charge for each time step
    model.z = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for charge/discharge interlocking
    model.v = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for curtailment/discharging interlocking

    # Initialize the parameters

    # Ensure that daa and pv_output are lists
    if not isinstance(daa, list):
        daa = daa.tolist()
    
    if not isinstance(pv_output, list):
        pv_output = pv_output.tolist()
    
    if not isinstance(market_premium, list):
        market_premium = market_premium.tolist()

    # Ensure the length of DAA and PV output matches the number of time steps
    if len(daa) != len(pv_output) or len(market_premium) != len(daa):
        raise ValueError("DAA and PV output data must have the same length.")

    # Initialize the constraints
    # Please note that the indices of price vectors and pv_input vector is t-1 while the model time index is t. 
    # This is because python lists are 0-indexed and pyomo uses 1-indexed time indices.

    ##### State of charge rules #####
    def state_of_charge_rule(model, t):
        return model.soc[t + 1] == model.soc[t] + model.p_charge_daa[t] * 0.25 * efficiency - model.p_discharge_daa[t] * 0.25 / efficiency 
    model.state_of_charge_contraint = pyo.Constraint(model.T, rule=state_of_charge_rule)

    # def minimum_soc_rule(model, t):
    #     return model.soc[t] >= 0.0  # Ensure state of charge is non-negative
    # model.minimum_soc_constraint = pyo.Constraint(model.T_plus_1, rule=minimum_soc_rule)

    def maximum_soc_rule(model, t):
        return model.soc[t] <= storage_capacity  # Ensure state of charge does not exceed capacity
    model.maximum_soc_constraint = pyo.Constraint(model.T_plus_1, rule=maximum_soc_rule)

    def charge_cycle_limit_rule(model):
        return sum(model.p_charge_daa[t] * 0.25 for t in model.T) <= number_of_cycles * storage_capacity
    model.cycle_charge_limit_constraint = pyo.Constraint(rule=charge_cycle_limit_rule)

    def start_soc_rule(model):
        return model.soc[1] == start_soc
    model.start_soc_constraint = pyo.Constraint(rule=start_soc_rule)

    def end_soc_rule(model):
        return model.soc[num_time_steps + 1] == end_soc
    model.end_soc_constraint = pyo.Constraint(rule=end_soc_rule)

    ##### Injection power rules #####
    def minium_injection_power_rule(model, t):
        return model.p_discharge_daa[t] + pv_output[t-1] - model.p_curtailed_daa[t] - model.p_charge_daa[t] >= 0
    model.minium_injection_power_constraint = pyo.Constraint(model.T, rule=minium_injection_power_rule)

    def maxium_injection_power_rule(model, t):
        return model.p_discharge_daa[t] + pv_output[t-1] - model.p_curtailed_daa[t] - model.p_charge_daa[t] <= p_limit
    model.maxium_injection_power_constraint = pyo.Constraint(model.T, rule=maxium_injection_power_rule)

    ##### Charge and discharge rules #####
    def p_charge_max_rule(model, t):
        return model.p_charge_daa[t] <= p_charge_max
    model.p_charge_max_constraint = pyo.Constraint(model.T, rule=p_charge_max_rule)

    def p_discharge_max_rule(model, t):  
        return model.p_discharge_daa[t] <= p_discharge_max
    model.p_discharge_max_constraint = pyo.Constraint(model.T, rule=p_discharge_max_rule)

    ##### Curtailment rules #####
    def curtailment_max_rule(model, t):
        return model.p_curtailed_daa[t] <= pv_output[t-1]
    model.curtailment_max_constraint = pyo.Constraint(model.T, rule=curtailment_max_rule)

    ##### Interlocking rules #####
    # The following two constraints prevents the optimizer from charging and discharging at the same time
    def p_charge_discharge_interlocking_rule(model, t):
        return model.p_charge_daa[t] <= p_charge_max * model.z[t]
    model.p_charge_discharge_interlocking_constraint = pyo.Constraint(model.T, rule=p_charge_discharge_interlocking_rule)
    
    def p_discharge_charge_interlocking_rule(model, t):
        return model.p_discharge_daa[t] <= p_discharge_max * (1-model.z[t])
    model.p_discharge_charge_interlocking_constraint = pyo.Constraint(model.T, rule=p_discharge_charge_interlocking_rule)

    # The following two constraints prevents the optimizer from curtailing and discharging at the same time
    def p_curtail_discharge_interlocking_rule(model, t):
        return model.p_curtailed_daa[t] <= pv_output[t-1] * (1-model.v[t])
    model.p_curtail_discharge_interlocking_constraint = pyo.Constraint(model.T, rule=p_curtail_discharge_interlocking_rule)

    def p_discharge_curtail_interlocking_rule(model, t):
        return model.p_discharge_daa[t] <= p_discharge_max * model.v[t]
    model.p_discharge_curtail_interlocking_constraint = pyo.Constraint(model.T, rule=p_discharge_curtail_interlocking_rule)

    ##### Power gradient rules #####
    def power_gradient_upper_rule(model, t):
        if t == 1:
            return pyo.Constraint.Skip
        return (model.p_charge_daa[t] - model.p_discharge_daa[t]) - (model.p_charge_daa[t-1] - model.p_discharge_daa[t-1]) <= delta_p_perm
    model.power_gradient_upper_constraint = pyo.Constraint(model.T, rule=power_gradient_upper_rule)

    def power_gradient_lower_rule(model, t):
        if t == 1:
            return pyo.Constraint.Skip
        return (model.p_charge_daa[t-1] - model.p_discharge_daa[t-1]) - (model.p_charge_daa[t] - model.p_discharge_daa[t]) <= delta_p_perm
    model.power_gradient_lower_constraint = pyo.Constraint(model.T, rule=power_gradient_lower_rule)
    
    ##### Injection power rule #####
    def injection_power_rule(model, t):
        return model.p_inject_daa[t] == model.p_discharge_daa[t] + pv_output[t-1] - model.p_curtailed_daa[t] - model.p_charge_daa[t]
    model.injection_power_constraint = pyo.Constraint(model.T, rule=injection_power_rule)

    ##### Objective function ##### 
    # Define the objective function
    model.objective = pyo.Objective(expr = sum((daa[t-1] + market_premium[t-1]) / 4 *  model.p_inject_daa[t] for t in model.T), sense=pyo.maximize)

    ##### Solve the model #####
    solver = pyo.SolverFactory(SOLVER)

    if PATH_TO_SOLVER:
        solver = pyo.SolverFactory(SOLVER, executable=PATH_TO_SOLVER)

    # Solve the model
    results = solver.solve(model, tee=True)

    # Check the results
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("Optimal solution found.")
    else:
        print("Solver did not find an optimal solution. Termination condition:", results.solver.termination_condition)

    # Extract the results to lists
    p_charge_daa = [model.p_charge_daa[t].value for t in range(1, len(daa) + 1)]
    p_discharge_daa = [model.p_discharge_daa[t].value for t in range(1, len(daa) + 1)]
    p_curtailed_daa = [model.p_curtailed_daa[t].value for t in range(1, len(daa) + 1)]
    p_inject_daa = [model.p_inject_daa[t].value for t in range(1, len(daa) + 1)]


    soc = [model.soc[t].value for t in range(1, len(daa) + 1)]

    return pv_output, daa, market_premium, p_charge_daa, p_discharge_daa, soc, p_curtailed_daa, p_inject_daa # injection_power


def get_ida_schedule(ida, market_premium, pv_output, p_limit, storage_capacity, p_charge_max, p_discharge_max,delta_p_perm, p_charge_daa, p_discharge_daa, p_curtailed_daa, number_of_cycles=2, efficiency=0.95, start_soc=0.0, end_soc=0.0):
    """
    Get the IDA schedule for the PV and BESS combination.
   
    Parameters:
    ----------
    - ida (list): List of IDA prices for each time step.
    - market_premium (list): list of market premia.
    - pv_output (list): The PV output for each time step.
    - p_limit (float): The power limit for the BESS.
    - storage_capacity (float): The storage capacity of the BESS.
    - p_charge_max (float): The maximum charging power for the BESS.
    - p_discharge_max (float): The maximum discharging power for the BESS.
    - delta_p_perm (float): Permissible change in power per time step.
    - p_charge_daa (list): The DAA charges for the BESS.
    - p_discharge_daa (list): The DAA discharges for the BESS.
    - p_curtailed_daa (list): The power curtailments for PV from DAA schedule.
    - number_of_cycles (int): The number of charge/discharge cycles (default is 2).
    - efficiency (float): The efficiency of the BESS (default is 0.95).
    - start_soc (float): The initial state of charge (default is 0.0).
    - end_soc (float): The final state of charge (default is 0.0).
    
    Returns:
    --------
    - ida: List of IDA prices for each time step.
    - p_charge_ida: List of IDA charging power for each time step.
    - p_discharge_ida: List of IDAdischarging power for each time step.
    - p_close_charge_daa: List of closed DAA charges for each time step.
    - p_close_discharge_daa: List of closed DAA discharges for each time step.
    - p_close_curtailed: List of closed curtailments for each time step.
    - p_curtailed_daa_ida: List of curtailed power for each time step.
    - p_charge_daa_ida: List of charging power for each time step.
    - p_discharge_daa_ida: List of discharging power for each time step.
    - soc_ida: List of state of charge for each time step.
    - injection_power_ida: List of injection power for each time step.
    """
    # Get the number of time steps
    num_time_steps = len(ida)

    # Initialize the model
    model = pyo.ConcreteModel()

    # Initialize the time parameters
    model.T = pyo.RangeSet(1, num_time_steps)  # Time step in 1/4 hour
    model.T_plus_1 = pyo.RangeSet(1, num_time_steps + 1)  # Time step in 1/4 hour plus one for state of chargeimport pyomo.environ as pyo

    # Initialize the vectorized variables with model.T elements
    model.p_charge_ida = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_discharge_ida = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_curtailed_ida = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_close_charge_daa = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_close_discharge_daa = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_close_curtailed_daa = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T_plus_1, within=pyo.NonNegativeReals, initialize=0.0)  # State of charge for each time step
    model.p_inject_ida = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_charge_daa_ida = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_discharge_daa_ida = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.z = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for charge/discharge decision (interlocking)
    model.y = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for close_discharge/discharge decision (interlocking)
    model.x = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for close_charge/charge decision (interlocking)
    model.w = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for close_curtailment/curtailment decision (interlocking)
    model.v = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for curtailment/discharging interlocking

    # Initialize the parameters

    # Ensure that daa and pv_output are lists
    if not isinstance(ida, list):
        ida = ida.tolist()

    if not isinstance(pv_output, list):
        pv_output = pv_output.tolist()

    # Ensure the length of DAA and PV output matches the number of time steps
    if len(ida) != len(pv_output):
        raise ValueError("IDA and PV output data must have the same length.")
   
    # Please note that the indices of price vectors and pv_input vector is t-1 while the model time index is t. 
    # This is because python lists are 0-indexed and pyomo uses 1-indexed time indices.

    # Initialize the constraints
    ##### State of charge rules #####
    def state_of_charge_rule(model, t):
        return model.soc[t + 1] == model.soc[t] + model.p_charge_ida[t] * 0.25 * efficiency - model.p_discharge_ida[t] * 0.25 / efficiency \
            - model.p_close_charge_daa[t] * 0.25 * efficiency + model.p_close_discharge_daa[t] * 0.25 / efficiency  \
            + p_charge_daa[t-1] * 0.25 * efficiency - p_discharge_daa[t-1] * 0.25 / efficiency
    model.state_of_charge_constraint = pyo.Constraint(model.T, rule=state_of_charge_rule)  # TODO Question: Is p_close_curtailed_daa part of the SOC equation?

    def maximum_soc_rule(model, t):
        return model.soc[t] <= storage_capacity  # Ensure state of charge does not exceed capacity
    model.maximum_soc_constraint = pyo.Constraint(model.T_plus_1, rule=maximum_soc_rule)

    def charge_cycle_limit_rule(model):
        return sum((p_charge_daa[t-1] + model.p_charge_ida[t] - model.p_close_charge_daa[t]) * 0.25 for t in model.T) <= number_of_cycles * storage_capacity
    model.cycle_charge_limit_constraint = pyo.Constraint(rule=charge_cycle_limit_rule)
    
    def start_soc_rule(model):
        return model.soc[1] == start_soc
    model.start_soc_constraint = pyo.Constraint(rule=start_soc_rule)

    def end_soc_rule(model):
        return model.soc[num_time_steps + 1] == end_soc
    model.end_soc_constraint = pyo.Constraint(rule=end_soc_rule)

    ##### Injection power rules #####
    def minium_injection_power_rule(model, t):
        return model.p_discharge_ida[t] - model.p_charge_ida[t] - model.p_curtailed_ida[t] \
            + p_discharge_daa[t-1] - p_charge_daa[t-1] \
            + model.p_close_charge_daa[t]- model.p_close_discharge_daa[t] \
            + model.p_close_curtailed_daa[t] \
            + pv_output[t-1] - p_curtailed_daa[t-1] >= 0
    model.minium_injection_power_constraint = pyo.Constraint(model.T, rule=minium_injection_power_rule)

    def maxium_injection_power_rule(model, t):
        return model.p_discharge_ida[t] - model.p_charge_ida[t] - model.p_curtailed_ida[t] \
            + p_discharge_daa[t-1] - p_charge_daa[t-1] \
            + model.p_close_charge_daa[t]- model.p_close_discharge_daa[t] \
            + model.p_close_curtailed_daa[t] \
            + pv_output[t-1] - p_curtailed_daa[t-1] <= p_limit
    model.maxium_injection_power_constraint = pyo.Constraint(model.T, rule=maxium_injection_power_rule)

    ##### Charge and discharge rules #####
    def p_charge_max_rule(model, t):
        return model.p_charge_ida[t] - model.p_close_charge_daa[t] + p_charge_daa[t-1] <= p_charge_max
    model.p_charge_max_constraint = pyo.Constraint(model.T, rule=p_charge_max_rule)

    def p_charge_min_rule(model, t):
        return model.p_charge_ida[t] - model.p_close_charge_daa[t] + p_charge_daa[t-1] >= 0.0
    model.p_charge_min_constraint = pyo.Constraint(model.T, rule=p_charge_min_rule)

    def p_discharge_max_rule(model, t):
        return model.p_discharge_ida[t] - model.p_close_discharge_daa[t] + p_discharge_daa[t-1] <= p_discharge_max
    model.p_discharge_max_constraint = pyo.Constraint(model.T, rule=p_discharge_max_rule)

    def p_discharge_min_rule(model, t):
        return model.p_discharge_ida[t] - model.p_close_discharge_daa[t] + p_discharge_daa[t-1] >= 0.0
    model.p_discharge_min_constraint = pyo.Constraint(model.T, rule=p_discharge_min_rule)

    ##### Curtailment rules #####
    def curtailment_max_rule(model, t):
        return model.p_curtailed_ida[t] <= pv_output[t-1] - p_curtailed_daa[t-1] + model.p_close_curtailed_daa[t]
    model.curtailment_max_constraint = pyo.Constraint(model.T, rule=curtailment_max_rule)

    ##### Closing rules #####
    def close_curtailment_max_rule(model, t):
        return model.p_close_curtailed_daa[t] <= p_curtailed_daa[t-1] * (1 - model.w[t]) # Use binary variable for interlocking TODO Maybe in interlocking in separate constraint
    model.close_curtailment_max_constraint = pyo.Constraint(model.T, rule=close_curtailment_max_rule)

    def close_daa_discharge_max_rule(model, t):
        return model.p_close_discharge_daa[t] <= p_discharge_daa[t-1] * (1 - model.y[t]) # Use binary variable for interlocking TODO Maybe in interlocking in separate constraint
    model.close_daa_discharge_max_constraint = pyo.Constraint(model.T, rule=close_daa_discharge_max_rule)

    def close_daa_charge_max_rule(model, t):
        return model.p_close_charge_daa[t] <= p_charge_daa[t-1] * (1 - model.x[t]) # Use binary variable for interlocking TODO Maybe in interlocking in separate constraint
    model.close_daa_charge_max_constraint = pyo.Constraint(model.T, rule=close_daa_charge_max_rule)

    ##### Interlocking rules #####
    # The following two constraints prevents the optimizer from charging and discharging the same asset at the same time
    def p_charge_discharge_interlocking_rule(model, t):
        return model.p_charge_ida[t] <= p_charge_max * model.z[t]
    model.p_charge_discharge_interlocking_constraint = pyo.Constraint(model.T, rule=p_charge_discharge_interlocking_rule)
    
    def p_discharge_charge_interlocking_rule(model, t):
        return model.p_discharge_ida[t] <= p_discharge_max * (1 - model.z[t])
    model.p_discharge_charge_interlocking_constraint = pyo.Constraint(model.T, rule=p_discharge_charge_interlocking_rule)

    # This rule prevents the solver from closing a DAA charge and renewing it on the IDA market (interlocking of closing and opening
    # the same asset at the same time)
    def p_charge_interlocking_rule(model, t):
        return model.p_charge_ida[t] <= p_charge_max * model.x[t]
    model.p_charge_interlocking_constraint = pyo.Constraint(model.T, rule=p_charge_interlocking_rule)

    # This rule prevents the solver from closing a DAA discharge and renewing it on the IDA market (interlocking of closing and opening
    # the same asset at the same time)
    def p_discharge_interlocking_rule(model, t):
        return model.p_discharge_ida[t] <= p_discharge_max * model.y[t]
    model.p_discharge_interlocking_constraint = pyo.Constraint(model.T, rule=p_discharge_interlocking_rule)

    # This rule prevents the solver from closing a DAA curtailment and renewing it on the IDA market (interlocking of closing and opening the same asset at the same time)
    def p_curtailment_interlocking_rule(model, t):
        return model.p_curtailed_ida[t] <= (pv_output[t-1] - p_curtailed_daa[t-1]) * model.w[t] # TODO Check if - p_curtailed_daa[t-1] is necessary
    model.p_curtailment_interlocking_constraint = pyo.Constraint(model.T, rule=p_curtailment_interlocking_rule)

    # The following two constraints prevents the optimizer from curtailing and discharging at the same time
    def p_curtail_discharge_interlocking_rule(model, t):
        return model.p_curtailed_ida[t] <= pv_output[t-1] * (1 - model.v[t])
    model.p_curtail_discharge_interlocking_constraint = pyo.Constraint(model.T, rule=p_curtail_discharge_interlocking_rule)

    def p_discharge_curtail_interlocking_rule(model, t):
        return model.p_discharge_ida[t] <= p_discharge_max * model.v[t]
    model.p_discharge_curtail_interlocking_constraint = pyo.Constraint(model.T, rule=p_discharge_curtail_interlocking_rule)

    #### Injection power rule #####
    def injection_power_rule(model, t):
        return model.p_inject_ida[t] == model.p_discharge_ida[t] - model.p_charge_ida[t] \
                       + p_discharge_daa[t-1] - p_charge_daa[t-1] \
                       + model.p_close_charge_daa[t] - model.p_close_discharge_daa[t] \
                       + pv_output[t-1] \
                       - p_curtailed_daa[t-1] + model.p_close_curtailed_daa[t] \
                       - model.p_curtailed_ida[t]
    model.injection_power_constraint = pyo.Constraint(model.T, rule=injection_power_rule)

    ##### Physical charge and discharge rules #####
    #  Physical charge rule
    def physical_charge_rule(model, t):
        return model.p_charge_ida[t] - model.p_close_charge_daa[t] + p_charge_daa[t-1] == model.p_charge_daa_ida[t]
    model.physical_charge_constraint = pyo.Constraint(model.T, rule=physical_charge_rule)

    # Physical discharge rule
    def physical_discharge_rule(model, t):
        return model.p_discharge_ida[t] - model.p_close_discharge_daa[t] + p_discharge_daa[t-1] == model.p_discharge_daa_ida[t]
    model.physical_discharge_constraint = pyo.Constraint(model.T, rule=physical_discharge_rule)

    ##### Power gradient rules #####
    def power_gradient_upper_rule(model, t):
        if t == 1:
            return pyo.Constraint.Skip
        return (model.p_charge_daa_ida[t] - model.p_discharge_daa_ida[t]) - (model.p_charge_daa_ida[t-1] - model.p_discharge_daa_ida[t-1]) <= delta_p_perm
    model.power_gradient_upper_constraint = pyo.Constraint(model.T, rule=power_gradient_upper_rule)

    def power_gradient_lower_rule(model, t):
        if t == 1:
            return pyo.Constraint.Skip
        return (model.p_charge_daa_ida[t-1] - model.p_discharge_daa_ida[t-1]) - (model.p_charge_daa_ida[t] - model.p_discharge_daa_ida[t]) <= delta_p_perm
    model.power_gradient_lower_constraint = pyo.Constraint(model.T, rule=power_gradient_lower_rule)
            
    ##### Objective function #####
    # Define the objective function
    model.objective = pyo.Objective(expr = sum(ida[t-1] / 4 * (model.p_discharge_ida[t]
                                                            - model.p_charge_ida[t]
                                                            + model.p_close_charge_daa[t]
                                                            - model.p_close_discharge_daa[t]
                                                            + model.p_close_curtailed_daa[t]
                                                            - model.p_curtailed_ida[t]) +  market_premium[t-1] / 4 * model.p_inject_ida[t] for t in model.T), sense=pyo.maximize)                              

    ##### Solve the model #####
    solver = pyo.SolverFactory(SOLVER)

    if PATH_TO_SOLVER:
        solver = pyo.SolverFactory(SOLVER, executable=PATH_TO_SOLVER)

    # Solve the model
    results = solver.solve(model, tee=True)

    # Check the results
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("Optimal solution found.")
    else:
        print("Solver did not find an optimal solution. Termination condition:", results.solver.termination_condition)

    # Extract the results to lists
    p_charge_ida = [model.p_charge_ida[t].value for t in range(1, len(ida) + 1)]
    p_discharge_ida = [model.p_discharge_ida[t].value for t in range(1, len(ida) + 1)]
    p_curtailed_ida = [model.p_curtailed_ida[t].value for t in range(1, len(ida) + 1)]
    p_close_charge_daa = [model.p_close_charge_daa[t].value for t in range(1, len(ida) + 1)]
    p_close_discharge_daa = [model.p_close_discharge_daa[t].value for t in range(1, len(ida) + 1)]
    p_close_curtailed_daa = [model.p_close_curtailed_daa[t].value for t in range(1, len(ida) + 1)]
    soc_ida = [model.soc[t].value for t in range(1, len(ida) + 1)]
    p_inject_ida = [model.p_inject_ida[t].value for t in range(1, len(ida) + 1)]

    # Calculate total physical charges/discharges and curtailments
    p_charge_daa_ida = [p_charge_ida[t] - p_close_charge_daa[t] + p_charge_daa[t] for t in range(0, len(ida))]
    p_discharge_daa_ida = [p_discharge_ida[t] - p_close_discharge_daa[t] + p_discharge_daa[t] for t in range(0, len(ida))]
    p_curtailed_daa_ida = [p_curtailed_daa[t] - p_close_curtailed_daa[t] + p_curtailed_ida[t] for t in range(0, len(ida))]
    
    return ida, p_charge_ida, p_discharge_ida, p_close_charge_daa, p_close_discharge_daa, p_curtailed_ida, \
        p_close_curtailed_daa, p_curtailed_daa_ida, p_charge_daa_ida, p_discharge_daa_ida, soc_ida, p_inject_ida # injection_power_ida

def get_idc_schedule(idc, market_premium, pv_output, p_limit, storage_capacity, p_charge_max, p_discharge_max, delta_p_perm, p_charge_daa_ida, p_discharge_daa_ida, p_curtailed_daa_ida, number_of_cycles=2, efficiency=0.95, start_soc=0.0, end_soc=0.0):
    """
    Get the IDC schedule for the PV and BESS combination.

    Parameter:
    ---------
    - idc (list): List of IDC prices for each time step.
    - market_premium (list): list of market premia.
    - pv_output (list): List of PV output values for each time step.
    - p_limit (float): Power limit for charging/discharging.
    - storage_capacity (float): Storage capacity of the battery.
    - p_charge_max (float): Maximum charging power.
    - p_discharge_max (float): Maximum discharging power.
    - delta_p_perm (float): Permissible change in power per time step.
    - p_charge_daa_ida (list): List of DAA charging power for each time step.
    - p_discharge_daa_ida (list): List of DAA discharging power for each time step.
    - p_curtailed_daa_ida (list): List of curtailment according to IDA schedule curtailed for each time step.
    - number_of_cycles (int): Number of charge/discharge cycles.
    - efficiency (float): Efficiency of the battery.
    - start_soc (float): Starting state of charge.
    - end_soc (float): Ending state of charge.

    Returns:
    --------
    - idc (list): List of IDC prices for each time step.
    - p_charge_idc (list): List of IDC charging power for each time step.
    - p_discharge_idc (list): List of IDC discharging power for each time step.
    - p_close_charge_daa_ida (list): List of combined closed DAA and IDA trades for each time step.
    - p_close_discharge_daa_ida (list): List of combined closed DAA and IDA trades for each time step.
    - p_close_curtailed (list): List of closed curtailments for each time step.
    - p_curtailed_daa_ida (list): List of curtailments according to IDA schedule for each time step.
    - p_charge_daa_ida (list): List with sums of DAA and IDA charges for each time step.
    - p_discharge_daa_ida (list): List with sums of DAA and IDA discharges for each time step.
    - soc_ida (list): List of state of charge according to IDA schedule for each time step.
    - injection_power_ida (list): List of injection power according to IDA schedule for each time step.
    """
    
    # Get the number of time steps
    num_time_steps = len(idc)

    # Initialize the model
    model = pyo.ConcreteModel()

    # Initialize the time parameters
    model.T = pyo.RangeSet(1, num_time_steps)  # Time step in 1/4 hour
    model.T_plus_1 = pyo.RangeSet(1, num_time_steps + 1)  # Time step in 1/4 hour plus one for state of chargeimport pyomo.environ as pyo

    # Initialize the vectorized variables with model.T elements
    model.p_charge_idc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_discharge_idc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_curtailed_idc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_close_charge_daa_ida = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_close_discharge_daa_ida = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_close_curtailed_daa_ida = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T_plus_1, within=pyo.NonNegativeReals, initialize=0.0)  # State of charge for each time step
    model.p_inject_idc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_charge_daa_ida_idc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.p_discharge_daa_ida_idc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.z = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for charge/discharge decision (interlocking)
    model.y = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for close_discharge/discharge decision (interlocking)
    model.x = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for close_charge/charge decision (interlocking)
    model.w = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for close_curtailment/curtailment decision (interlocking)
    model.v = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for curtailment/discharging interlocking

    # Initialize the parameters

    # Ensure that daa and pv_output are lists
    if not isinstance(idc, list):
        idc = idc.tolist()

    if not isinstance(pv_output, list):
        pv_output = pv_output.tolist()

    # Ensure the length of DAA and PV output matches the number of time steps
    if len(idc) != len(pv_output):
        raise ValueError("IDC and PV output data must have the same length.")
   
    # Please note that the indices of price vectors and pv_input vector is t-1 while the model time index is t. 
    # This is because python lists are 0-indexed and pyomo uses 1-indexed time indices.

    # Initialize the constraints
    ##### State of charge rules #####
    def state_of_charge_rule(model, t):
        return model.soc[t + 1] == model.soc[t] + model.p_charge_idc[t] * 0.25 * efficiency - model.p_discharge_idc[t] * 0.25 / efficiency \
            - model.p_close_charge_daa_ida[t] * 0.25 * efficiency + model.p_close_discharge_daa_ida[t] * 0.25 / efficiency  \
            + p_charge_daa_ida[t-1] * 0.25 * efficiency - p_discharge_daa_ida[t-1] * 0.25 / efficiency
    model.state_of_charge_constraint = pyo.Constraint(model.T, rule=state_of_charge_rule)  # TODO Question: Is p_close_curtailed_daa_ida part of the SOC equation?

    def start_soc_rule(model):
        return model.soc[1] == start_soc
    model.start_soc_constraint = pyo.Constraint(rule=start_soc_rule)

    def end_soc_rule(model):
        return model.soc[num_time_steps + 1] == end_soc
    model.end_soc_constraint = pyo.Constraint(rule=end_soc_rule)

    def maximum_soc_rule(model, t):
        return model.soc[t] <= storage_capacity  # Ensure state of charge does not exceed capacity
    model.maximum_soc_constraint = pyo.Constraint(model.T_plus_1, rule=maximum_soc_rule)

    def charge_cycle_limit_rule(model):
        return sum((p_charge_daa_ida[t-1] + model.p_charge_idc[t] - model.p_close_charge_daa_ida[t]) * 0.25 for t in model.T) <= number_of_cycles * storage_capacity
    model.cycle_charge_limit_constraint = pyo.Constraint(rule=charge_cycle_limit_rule)

    ##### Injection power rules #####
    def minium_injection_power_rule(model, t):
        return model.p_discharge_idc[t] - model.p_charge_idc[t] - model.p_curtailed_idc[t] \
            + p_discharge_daa_ida[t-1] - p_charge_daa_ida[t-1] \
            + model.p_close_charge_daa_ida[t]- model.p_close_discharge_daa_ida[t] \
            + model.p_close_curtailed_daa_ida[t] \
            + pv_output[t-1] - p_curtailed_daa_ida[t-1] >= 0
    model.minium_injection_power_constraint = pyo.Constraint(model.T, rule=minium_injection_power_rule)

    def maxium_injection_power_rule(model, t):
        return model.p_discharge_idc[t] - model.p_charge_idc[t] - model.p_curtailed_idc[t] \
            + p_discharge_daa_ida[t-1] - p_charge_daa_ida[t-1] \
            + model.p_close_charge_daa_ida[t]- model.p_close_discharge_daa_ida[t] \
            + model.p_close_curtailed_daa_ida[t] \
            + pv_output[t-1] - p_curtailed_daa_ida[t-1] <= p_limit
    model.maxium_injection_power_constraint = pyo.Constraint(model.T, rule=maxium_injection_power_rule)

    ##### Charge and discharge rules #####
    def p_charge_max_rule(model, t):
        return model.p_charge_idc[t] - model.p_close_charge_daa_ida[t] + p_charge_daa_ida[t-1] <= p_charge_max
    model.p_charge_max_constraint = pyo.Constraint(model.T, rule=p_charge_max_rule)

    def p_charge_min_rule(model, t):
        return model.p_charge_idc[t] - model.p_close_charge_daa_ida[t] + p_charge_daa_ida[t-1] >= 0.0
    model.p_charge_min_constraint = pyo.Constraint(model.T, rule=p_charge_min_rule)

    def p_discharge_max_rule(model, t):
        return model.p_discharge_idc[t] - model.p_close_discharge_daa_ida[t] + p_discharge_daa_ida[t-1] <= p_discharge_max
    model.p_discharge_max_constraint = pyo.Constraint(model.T, rule=p_discharge_max_rule)

    def p_discharge_min_rule(model, t):
        return model.p_discharge_idc[t] - model.p_close_discharge_daa_ida[t] + p_discharge_daa_ida[t-1] >= 0.0
    model.p_discharge_min_constraint = pyo.Constraint(model.T, rule=p_discharge_min_rule)

    ##### Curtailment rules #####
    def curtailment_max_rule(model, t):
        return model.p_curtailed_idc[t] <= pv_output[t-1] - p_curtailed_daa_ida[t-1] + model.p_close_curtailed_daa_ida[t]
    model.curtailment_max_constraint = pyo.Constraint(model.T, rule=curtailment_max_rule)

    ##### Closing rules #####
    def close_curtailment_max_rule(model, t):
        return model.p_close_curtailed_daa_ida[t] <= p_curtailed_daa_ida[t-1] * (1 - model.w[t])  # Use binary variable for interlocking TODO Maybe in interlocking in separate constraint
    model.close_curtailment_max_constraint = pyo.Constraint(model.T, rule=close_curtailment_max_rule)

    def close_daa_discharge_max_rule(model, t):
        return model.p_close_discharge_daa_ida[t] <= p_discharge_daa_ida[t-1] * (1 - model.y[t]) # Use binary variable for interlocking TODO Maybe in interlocking in separate constraint
    model.close_daa_discharge_max_constraint = pyo.Constraint(model.T, rule=close_daa_discharge_max_rule)

    def close_daa_charge_max_rule(model, t):
        return model.p_close_charge_daa_ida[t] <= p_charge_daa_ida[t-1] * (1 - model.x[t]) # Use binary variable for interlocking TODO Maybe in interlocking in separate constraint
    model.close_daa_charge_max_constraint = pyo.Constraint(model.T, rule=close_daa_charge_max_rule)

    ##### Interlocking rules #####
    # The following two constraints prevents the optimizer from charging and discharging the same asset at the same time
    def p_charge_discharge_interlocking_rule(model, t):
        return model.p_charge_idc[t] <= p_charge_max * model.z[t]
    model.p_charge_discharge_interlocking_constraint = pyo.Constraint(model.T, rule=p_charge_discharge_interlocking_rule)
    
    def p_discharge_charge_interlocking_rule(model, t):
        return model.p_discharge_idc[t] <= p_discharge_max * (1 - model.z[t])
    model.p_discharge_charge_interlocking_constraint = pyo.Constraint(model.T, rule=p_discharge_charge_interlocking_rule)

    # This rule prevents the solver from closing a DAA charge and renewing it on the IDA market (interlocking of closing and opening 
    # the same asset at the same time)
    def p_charge_interlocking_rule(model, t):
        return model.p_charge_idc[t] <= p_charge_max * model.x[t]
    model.p_charge_interlocking_constraint = pyo.Constraint(model.T, rule=p_charge_interlocking_rule)

    # This rule prevents the solver from closing a DAA discharge and renewing it on the IDA market (interlocking of closing and opening 
    # the same asset at the same time)
    def p_discharge_interlocking_rule(model, t):
        return model.p_discharge_idc[t] <= p_discharge_max * model.y[t]
    model.p_discharge_interlocking_constraint = pyo.Constraint(model.T, rule=p_discharge_interlocking_rule)
 
    # This rule prevents the solver from closing a DAA IDA curtailment and renewing it on the IDC market (interlocking of closing and opening 
    # the same asset at the same time)
    def p_curtailment_interlocking_rule(model, t):
        return model.p_curtailed_idc[t] <= (pv_output[t-1] - p_curtailed_daa_ida[t-1]) * model.w[t] # TODO check if p_curtailed_daa_ida is necessary here
    model.p_curtailment_interlocking_constraint = pyo.Constraint(model.T, rule=p_curtailment_interlocking_rule)

    # The following two constraints prevents the optimizer from curtailing and discharging at the same time
    def p_curtail_discharge_interlocking_rule(model, t):
        return model.p_curtailed_idc[t] <= pv_output[t - 1] * (1 - model.v[t])
    model.p_curtail_discharge_interlocking_constraint = pyo.Constraint(model.T, rule=p_curtail_discharge_interlocking_rule)

    def p_discharge_curtail_interlocking_rule(model, t):
        return model.p_discharge_idc[t] <= p_discharge_max * model.v[t]
    model.p_discharge_curtail_interlocking_constraint = pyo.Constraint(model.T, rule=p_discharge_curtail_interlocking_rule)

    ##### Injection power rule #####
    def injection_power_rule(model, t):
        return model.p_inject_idc[t] == model.p_discharge_idc[t] - model.p_charge_idc[t] \
                                            + p_discharge_daa_ida[t-1] - p_charge_daa_ida[t-1] \
                                            + model.p_close_charge_daa_ida[t] - model.p_close_discharge_daa_ida[t] \
                                            + pv_output[t-1] \
                                            - p_curtailed_daa_ida[t-1] + model.p_close_curtailed_daa_ida[t] \
                                            - model.p_curtailed_idc[t]
    model.injection_power_constraint = pyo.Constraint(model.T, rule=injection_power_rule)

    ##### Physical charge and discharge rules #####
    #  Physical charge rule
    def physical_charge_rule(model, t):
        return model.p_charge_idc[t] - model.p_close_charge_daa_ida[t] + p_charge_daa_ida[t-1] == model.p_charge_daa_ida_idc[t]
    model.physical_charge_constraint = pyo.Constraint(model.T, rule=physical_charge_rule)

    # Physical discharge rule
    def physical_discharge_rule(model, t):
        return model.p_discharge_idc[t] - model.p_close_discharge_daa_ida[t] + p_discharge_daa_ida[t-1] == model.p_discharge_daa_ida_idc[t]
    model.physical_discharge_constraint = pyo.Constraint(model.T, rule=physical_discharge_rule)
    
    ##### Power gradient rules #####
    def power_gradient_upper_rule(model, t):
        if t == 1:
            return pyo.Constraint.Skip
        return (model.p_charge_daa_ida_idc[t] - model.p_discharge_daa_ida_idc[t]) - (model.p_charge_daa_ida_idc[t-1] - model.p_discharge_daa_ida_idc[t-1]) <= delta_p_perm
    model.power_gradient_upper_constraint = pyo.Constraint(model.T, rule=power_gradient_upper_rule)

    def power_gradient_lower_rule(model, t):
        if t == 1:
            return pyo.Constraint.Skip
        return (model.p_charge_daa_ida_idc[t-1] - model.p_discharge_daa_ida_idc[t-1]) - (model.p_charge_daa_ida_idc[t] - model.p_discharge_daa_ida_idc[t]) <= delta_p_perm
    model.power_gradient_lower_constraint = pyo.Constraint(model.T, rule=power_gradient_lower_rule)

    ##### Objective function #####
    # Define the objective function
    model.objective = pyo.Objective(expr = sum(idc[t-1] / 4 * (model.p_discharge_idc[t]
                                                            - model.p_charge_idc[t]
                                                            + model.p_close_charge_daa_ida[t]
                                                            - model.p_close_discharge_daa_ida[t]
                                                            + model.p_close_curtailed_daa_ida[t]
                                                            - model.p_curtailed_idc[t]) +  market_premium[t-1] / 4 * model.p_inject_idc[t]for t in model.T), sense=pyo.maximize)
    ##### Solve the model #####
    solver = pyo.SolverFactory(SOLVER)

    if PATH_TO_SOLVER:
        solver = pyo.SolverFactory(SOLVER, executable=PATH_TO_SOLVER)

    # Solve the model
    results = solver.solve(model, tee=True)

    # Check the results
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("Optimal solution found.")
    else:
        print("Solver did not find an optimal solution. Termination condition:", results.solver.termination_condition)

    # Extract the results to lists
    p_charge_idc = [model.p_charge_idc[t].value for t in range(1, len(idc) + 1)]
    p_discharge_idc = [model.p_discharge_idc[t].value for t in range(1, len(idc) + 1)]
    p_curtailed_idc = [model.p_curtailed_idc[t].value for t in range(1, len(idc) + 1)]
    p_close_charge_daa_ida = [model.p_close_charge_daa_ida[t].value for t in range(1, len(idc) + 1)]
    p_close_discharge_daa_ida = [model.p_close_discharge_daa_ida[t].value for t in range(1, len(idc) + 1)]
    p_close_curtailed_daa_ida = [model.p_close_curtailed_daa_ida[t].value for t in range(1, len(idc) + 1)]
    soc_idc = [model.soc[t].value for t in range(1, len(idc) + 1)]
    p_inject_idc = [model.p_inject_idc[t].value for t in range(1, len(idc) + 1)]

    # Calculate total physical charges/discharges and curtailments
    p_charge_daa_ida_idc = [p_charge_idc[t] - p_close_charge_daa_ida[t] + p_charge_daa_ida[t] for t in range(0, len(idc))]
    p_discharge_daa_ida_idc = [p_discharge_idc[t] - p_close_discharge_daa_ida[t] + p_discharge_daa_ida[t] for t in range(0, len(idc))]
    p_curtailed_daa_ida_idc = [p_curtailed_daa_ida[t] - p_close_curtailed_daa_ida[t] + p_curtailed_idc[t] for t in range(0, len(idc))]

    return idc, p_charge_idc, p_discharge_idc, p_close_charge_daa_ida, p_close_discharge_daa_ida, p_curtailed_idc, \
        p_close_curtailed_daa_ida, p_curtailed_daa_ida_idc, p_charge_daa_ida_idc, p_discharge_daa_ida_idc, soc_idc, p_inject_idc


