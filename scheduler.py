
import pyomo.environ as pyo

# This module comprises a Pyomo optimization model to schedule the operation of a photovoltaic (PV) system combined 
# with a battery energy storage system (BESS).
# The model optimizes the charging and discharging of the BESS to maximize the economic benefit from the PV output and 
# DAA prices, while adhering to various constraints such as power limits, storage capacity, and efficiency.

# Pyomo requires a solver to be installed and specified.
# Ensure you have a solver installed, such as GLPK, CBC, or HiGHS.

# For example HiGHS via pip: pip install pyomo[solvers]

SOLVER = 'highs'  # Specify the solver to use, e.g., 'glpk', 'cbc', 'highs'
PATH_TO_SOLVER = None  # Optional: specify the path to the solver executable if needed


def get_schedule(pv_output, daa, p_limit, storage_capacity, p_charge_max, p_discharge_max, number_of_cycles=2, efficiency=0.95, start_soc=0.0, end_soc=0.0):
    """    
    Function to get the schedule for the PV and BESS combination.
    This function initializes the Pyomo model, sets up the variables, constraints, and objective function,
    and then solves the optimization problem.
    
    Parameters:
    ------------
    pv_output (list): List of PV output values for each time step.
    daa (list): List of DAA prices for each time step.
    p_limit (float): Maximum grid injection power limit.
    storage_capacity (float): Maximum storage capacity of the BESS.
    p_charge_max (float): Maximum charging power of the BESS.
    p_discharge_max (float): Maximum discharging power of the BESS.
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
    model.p_charge = pyo.Var(model.T, within=pyo.Reals)
    model.p_discharge = pyo.Var(model.T, within=pyo.Reals)
    model.p_curtailed = pyo.Var(model.T, within=pyo.Reals)
    model.soc = pyo.Var(model.T_plus_1, within=pyo.Reals, initialize=0.0)  # State of charge for each time step
    model.z = pyo.Var(model.T, within=pyo.Binary)  # Binary variable for charge/discharge decision

    # Initialize the parameters

    # Ensure that daa and pv_output are lists
    if not isinstance(daa, list):
        daa = daa.tolist()
    
    if not isinstance(pv_output, list):
        pv_output = pv_output.tolist()

    # Ensure the length of DAA and PV output matches the number of time steps
    if len(daa) != len(pv_output):
        raise ValueError("DAA and PV output data must have the same length.")

    # Initialize the constraints
    # Please note that the indices of DAA price vector and pv_input vector is t-1 while the model time index is t. 
    # This is because python lists are 0-indexed and pyomo uses 1-indexed time indices.
    def state_of_charge_rule(model, t):
        return model.soc[t + 1] == model.soc[t] + model.p_charge[t] * 0.25 * efficiency - model.p_discharge[t] * 0.25 / efficiency 
    model.state_of_charge_contraint = pyo.Constraint(model.T, rule=state_of_charge_rule)

    def start_soc_rule(model):
        return model.soc[1] == start_soc
    model.start_soc_constraint = pyo.Constraint(rule=start_soc_rule)

    def end_soc_rule(model):
        return model.soc[num_time_steps + 1] == end_soc
    model.end_soc_constraint = pyo.Constraint(rule=end_soc_rule)

    def minium_injection_power_rule(model, t):
        return model.p_discharge[t] + pv_output[t - 1] - model.p_curtailed[t] - model.p_charge[t] >= 0
    model.minium_injection_power_constraint = pyo.Constraint(model.T, rule=minium_injection_power_rule)

    def maxium_injection_power_rule(model, t):
        return model.p_discharge[t] + pv_output[t - 1] - model.p_curtailed[t] - model.p_charge[t] <= p_limit
    model.maxium_injection_power_constraint = pyo.Constraint(model.T, rule=maxium_injection_power_rule)

    def p_charge_max_rule(model, t):
        return model.p_charge[t] <= p_charge_max * model.z[t]  # Use binary variable to limit charge power
    model.p_charge_max_constraint = pyo.Constraint(model.T, rule=p_charge_max_rule)

    def p_charge_min_rule(model, t):
        return model.p_charge[t] >= 0.0
    model.p_charge_min_constraint = pyo.Constraint(model.T, rule=p_charge_min_rule)

    def p_discharge_max_rule(model, t):
        return model.p_discharge[t] <= p_discharge_max * (1 - model.z[t])  # Use binary variable to limit discharge power
    model.p_discharge_max_constraint = pyo.Constraint(model.T, rule=p_discharge_max_rule)

    def p_discharge_min_rule(model, t):
        return model.p_discharge[t] >= 0.0
    model.p_discharge_min_constraint = pyo.Constraint(model.T, rule=p_discharge_min_rule)

    def curtailment_max_rule(model, t):
        return model.p_curtailed[t] <= pv_output[t - 1]
    model.curtailment_max_constraint = pyo.Constraint(model.T, rule=curtailment_max_rule)

    def curtailment_min_rule(model, t):
        return model.p_curtailed[t] >= 0.0
    model.curtailment_min_constraint = pyo.Constraint(model.T, rule=curtailment_min_rule)

    def minimum_soc_rule(model, t):
        return model.soc[t] >= 0.0  # Ensure state of charge is non-negative
    model.minimum_soc_constraint = pyo.Constraint(model.T_plus_1, rule=minimum_soc_rule)

    def maximum_soc_rule(model, t):
        return model.soc[t] <= storage_capacity  # Ensure state of charge does not exceed capacity
    model.maximum_soc_constraint = pyo.Constraint(model.T_plus_1, rule=maximum_soc_rule)

    def charge_cycle_limit_rule(model):
        return sum(model.p_charge[t] * 0.25 for t in model.T) <= number_of_cycles * storage_capacity
    model.cycle_charge_limit_constraint = pyo.Constraint(rule=charge_cycle_limit_rule)

    # Define the objective function
    model.objective = pyo.Objective(expr = sum(daa[t-1] / 4 * (- model.p_charge[t]
                                                            + model.p_discharge[t] 
                                                            - model.p_curtailed[t] 
                                                            + pv_output[t-1]) for t in model.T), sense=pyo.maximize)

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
    p_charge = [model.p_charge[t].value for t in range(1, len(daa) + 1)]
    p_discharge = [model.p_discharge[t].value for t in range(1, len(daa) + 1)]
    p_curtailed = [model.p_curtailed[t].value for t in range(1, len(daa) + 1)]
    soc = [model.soc[t].value for t in range(1, len(daa) + 1)]

    # Calculate the injection power
    # injection_power = [p_discharge[t] + pv_output[t-1] - p_curtailed[t] - p_charge[t] for t in range(0, len(daa))]
    injection_power = [p_discharge[t] + pv_output[t] - p_curtailed[t] - p_charge[t] for t in range(0, len(daa))]

    return pv_output, daa, p_charge, p_discharge, soc, p_curtailed, injection_power
