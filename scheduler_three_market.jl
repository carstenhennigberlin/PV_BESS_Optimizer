module Scheduler

using JuMP
using HiGHS
# using Cbc

const DEFAULT_TIME_STEP_HOURS = 0.25

export get_daa_schedule, get_ida_schedule, get_idc_schedule

to_float_vector(values) = Float64.(collect(values))

function validate_lengths(reference, vectors...; names=String[])
    expected_length = length(reference)
    for (index, vector) in enumerate(vectors)
        if length(vector) != expected_length
            label = index <= length(names) ? names[index] : "vector $(index)"
            throw(ArgumentError("Length mismatch for $(label): expected $(expected_length), got $(length(vector))."))
        end
    end
end

function optimize_and_check!(model::Model)
    optimize!(model)
    if !is_solved_and_feasible(model; dual=false)
        throw(ErrorException("Solver did not find an optimal solution. Termination status: $(termination_status(model))"))
    end
    return model
end

function get_daa_schedule(
    pv_output,
    daa,
    market_premium,
    p_limit,
    storage_capacity,
    p_charge_max,
    p_discharge_max,
    delta_p_perm,
    number_of_cycles::Integer=2,
    efficiency=0.95,
    start_soc=0.0,
    end_soc=0.0;
    optimizer=HiGHS.Optimizer,
)
    pv_output = to_float_vector(pv_output)
    daa = to_float_vector(daa)
    market_premium = to_float_vector(market_premium)
    validate_lengths(daa, pv_output, market_premium; names=["pv_output", "market_premium"])

    num_time_steps = length(daa)
    model = Model(optimizer)

    @variable(model, p_charge_daa[1:num_time_steps] >= 0)
    @variable(model, p_discharge_daa[1:num_time_steps] >= 0)
    @variable(model, p_curtailed_daa[1:num_time_steps] >= 0)
    @variable(model, p_inject_daa[1:num_time_steps] >= 0)
    @variable(model, soc[1:num_time_steps + 1] >= 0)
    @variable(model, z[1:num_time_steps], Bin)
    @variable(model, v[1:num_time_steps], Bin)

    @constraint(model, [t in 1:num_time_steps],
        soc[t + 1] == soc[t] + p_charge_daa[t] * DEFAULT_TIME_STEP_HOURS * efficiency - p_discharge_daa[t] * DEFAULT_TIME_STEP_HOURS / efficiency)
    @constraint(model, [t in 1:num_time_steps + 1], soc[t] <= storage_capacity)
    @constraint(model,
        sum(p_charge_daa[t] * DEFAULT_TIME_STEP_HOURS for t in 1:num_time_steps) <= number_of_cycles * storage_capacity)
    @constraint(model, soc[1] == start_soc)
    @constraint(model, soc[num_time_steps + 1] == end_soc)

    @constraint(model, [t in 1:num_time_steps],
        p_discharge_daa[t] + pv_output[t] - p_curtailed_daa[t] - p_charge_daa[t] >= 0)
    @constraint(model, [t in 1:num_time_steps],
        p_discharge_daa[t] + pv_output[t] - p_curtailed_daa[t] - p_charge_daa[t] <= p_limit)

    @constraint(model, [t in 1:num_time_steps], p_charge_daa[t] <= p_charge_max)
    @constraint(model, [t in 1:num_time_steps], p_discharge_daa[t] <= p_discharge_max)
    @constraint(model, [t in 1:num_time_steps], p_curtailed_daa[t] <= pv_output[t])

    @constraint(model, [t in 1:num_time_steps], p_charge_daa[t] <= p_charge_max * z[t])
    @constraint(model, [t in 1:num_time_steps], p_discharge_daa[t] <= p_discharge_max * (1 - z[t]))
    @constraint(model, [t in 1:num_time_steps], p_curtailed_daa[t] <= pv_output[t] * (1 - v[t]))
    @constraint(model, [t in 1:num_time_steps], p_discharge_daa[t] <= p_discharge_max * v[t])

    @constraint(model, [t in 2:num_time_steps],
        (p_charge_daa[t] - p_discharge_daa[t]) - (p_charge_daa[t - 1] - p_discharge_daa[t - 1]) <= delta_p_perm)
    @constraint(model, [t in 2:num_time_steps],
        (p_charge_daa[t - 1] - p_discharge_daa[t - 1]) - (p_charge_daa[t] - p_discharge_daa[t]) <= delta_p_perm)

    @constraint(model, [t in 1:num_time_steps],
        p_inject_daa[t] == p_discharge_daa[t] + pv_output[t] - p_curtailed_daa[t] - p_charge_daa[t])

    @objective(model, Max,
        sum((daa[t] + market_premium[t]) * p_inject_daa[t] * DEFAULT_TIME_STEP_HOURS for t in 1:num_time_steps))

    optimize_and_check!(model)

    return (
        pv_output=pv_output,
        daa=daa,
        market_premium=market_premium,
        p_charge_daa=value.(p_charge_daa),
        p_discharge_daa=value.(p_discharge_daa),
        soc=value.(soc[1:num_time_steps]),
        p_curtailed_daa=value.(p_curtailed_daa),
        p_inject_daa=value.(p_inject_daa),
    )
end

function get_ida_schedule(
    ida,
    market_premium,
    pv_output,
    p_limit,
    storage_capacity,
    p_charge_max,
    p_discharge_max,
    delta_p_perm,
    p_charge_daa,
    p_discharge_daa,
    p_curtailed_daa,
    number_of_cycles::Integer=2,
    efficiency=0.95,
    start_soc=0.0,
    end_soc=0.0;
    optimizer=HiGHS.Optimizer,
    # optimizer=Cbc.Optimizer
)
    ida = to_float_vector(ida)
    market_premium = to_float_vector(market_premium)
    pv_output = to_float_vector(pv_output)
    p_charge_daa = to_float_vector(p_charge_daa)
    p_discharge_daa = to_float_vector(p_discharge_daa)
    p_curtailed_daa = to_float_vector(p_curtailed_daa)
    validate_lengths(
        ida,
        market_premium,
        pv_output,
        p_charge_daa,
        p_discharge_daa,
        p_curtailed_daa;
        names=["market_premium", "pv_output", "p_charge_daa", "p_discharge_daa", "p_curtailed_daa"],
    )

    num_time_steps = length(ida)
    model = Model(optimizer)

    @variable(model, p_charge_ida[1:num_time_steps] >= 0)
    @variable(model, p_discharge_ida[1:num_time_steps] >= 0)
    @variable(model, p_curtailed_ida[1:num_time_steps] >= 0)
    @variable(model, p_close_charge_daa[1:num_time_steps] >= 0)
    @variable(model, p_close_discharge_daa[1:num_time_steps] >= 0)
    @variable(model, p_close_curtailed_daa[1:num_time_steps] >= 0)
    @variable(model, soc[1:num_time_steps + 1] >= 0)
    @variable(model, p_inject_ida[1:num_time_steps] >= 0)
    @variable(model, p_charge_daa_ida[1:num_time_steps] >= 0)
    @variable(model, p_discharge_daa_ida[1:num_time_steps] >= 0)
    @variable(model, z[1:num_time_steps], Bin)
    @variable(model, y[1:num_time_steps], Bin)
    @variable(model, x[1:num_time_steps], Bin)
    @variable(model, w[1:num_time_steps], Bin)
    @variable(model, v[1:num_time_steps], Bin)

    @constraint(model, [t in 1:num_time_steps],
        soc[t + 1] == soc[t] + p_charge_ida[t] * DEFAULT_TIME_STEP_HOURS * efficiency - p_discharge_ida[t] * DEFAULT_TIME_STEP_HOURS / efficiency - p_close_charge_daa[t] * DEFAULT_TIME_STEP_HOURS * efficiency + p_close_discharge_daa[t] * DEFAULT_TIME_STEP_HOURS / efficiency + p_charge_daa[t] * DEFAULT_TIME_STEP_HOURS * efficiency - p_discharge_daa[t] * DEFAULT_TIME_STEP_HOURS / efficiency)
    @constraint(model, [t in 1:num_time_steps + 1], soc[t] <= storage_capacity)
    @constraint(model,
        sum((p_charge_daa[t] + p_charge_ida[t] - p_close_charge_daa[t]) * DEFAULT_TIME_STEP_HOURS for t in 1:num_time_steps) <= number_of_cycles * storage_capacity)
    @constraint(model, soc[1] == start_soc)
    @constraint(model, soc[num_time_steps + 1] == end_soc)

    @constraint(model, [t in 1:num_time_steps],
        p_discharge_ida[t] - p_charge_ida[t] - p_curtailed_ida[t] + p_discharge_daa[t] - p_charge_daa[t] + p_close_charge_daa[t] - p_close_discharge_daa[t] + p_close_curtailed_daa[t] + pv_output[t] - p_curtailed_daa[t] >= 0)
    @constraint(model, [t in 1:num_time_steps],
        p_discharge_ida[t] - p_charge_ida[t] - p_curtailed_ida[t] + p_discharge_daa[t] - p_charge_daa[t] + p_close_charge_daa[t] - p_close_discharge_daa[t] + p_close_curtailed_daa[t] + pv_output[t] - p_curtailed_daa[t] <= p_limit)

    @constraint(model, [t in 1:num_time_steps], p_charge_ida[t] - p_close_charge_daa[t] + p_charge_daa[t] <= p_charge_max)
    @constraint(model, [t in 1:num_time_steps], p_charge_ida[t] - p_close_charge_daa[t] + p_charge_daa[t] >= 0)
    @constraint(model, [t in 1:num_time_steps], p_discharge_ida[t] - p_close_discharge_daa[t] + p_discharge_daa[t] <= p_discharge_max)
    @constraint(model, [t in 1:num_time_steps], p_discharge_ida[t] - p_close_discharge_daa[t] + p_discharge_daa[t] >= 0)
    @constraint(model, [t in 1:num_time_steps], p_curtailed_ida[t] <= pv_output[t] - p_curtailed_daa[t] + p_close_curtailed_daa[t])

    @constraint(model, [t in 1:num_time_steps], p_close_curtailed_daa[t] <= p_curtailed_daa[t] * (1 - w[t]))
    @constraint(model, [t in 1:num_time_steps], p_close_discharge_daa[t] <= p_discharge_daa[t] * (1 - y[t]))
    @constraint(model, [t in 1:num_time_steps], p_close_charge_daa[t] <= p_charge_daa[t] * (1 - x[t]))

    @constraint(model, [t in 1:num_time_steps], p_charge_ida[t] <= p_charge_max * z[t])
    @constraint(model, [t in 1:num_time_steps], p_discharge_ida[t] <= p_discharge_max * (1 - z[t]))
    @constraint(model, [t in 1:num_time_steps], p_charge_ida[t] <= p_charge_max * x[t])
    @constraint(model, [t in 1:num_time_steps], p_discharge_ida[t] <= p_discharge_max * y[t])
    @constraint(model, [t in 1:num_time_steps], p_curtailed_ida[t] <= (pv_output[t] - p_curtailed_daa[t]) * w[t])
    @constraint(model, [t in 1:num_time_steps], p_curtailed_ida[t] <= pv_output[t] * (1 - v[t]))
    @constraint(model, [t in 1:num_time_steps], p_discharge_ida[t] <= p_discharge_max * v[t])

    @constraint(model, [t in 1:num_time_steps],
        p_inject_ida[t] == p_discharge_ida[t] - p_charge_ida[t] + p_discharge_daa[t] - p_charge_daa[t] + p_close_charge_daa[t] - p_close_discharge_daa[t] + pv_output[t] - p_curtailed_daa[t] + p_close_curtailed_daa[t] - p_curtailed_ida[t])
    @constraint(model, [t in 1:num_time_steps], p_charge_ida[t] - p_close_charge_daa[t] + p_charge_daa[t] == p_charge_daa_ida[t])
    @constraint(model, [t in 1:num_time_steps], p_discharge_ida[t] - p_close_discharge_daa[t] + p_discharge_daa[t] == p_discharge_daa_ida[t])

    @constraint(model, [t in 2:num_time_steps],
        (p_charge_daa_ida[t] - p_discharge_daa_ida[t]) - (p_charge_daa_ida[t - 1] - p_discharge_daa_ida[t - 1]) <= delta_p_perm)
    @constraint(model, [t in 2:num_time_steps],
        (p_charge_daa_ida[t - 1] - p_discharge_daa_ida[t - 1]) - (p_charge_daa_ida[t] - p_discharge_daa_ida[t]) <= delta_p_perm)

    @objective(model, Max,
        sum(ida[t] * (p_discharge_ida[t] - p_charge_ida[t] + p_close_charge_daa[t] - p_close_discharge_daa[t] + p_close_curtailed_daa[t] - p_curtailed_ida[t]) * DEFAULT_TIME_STEP_HOURS + market_premium[t] * p_inject_ida[t] * DEFAULT_TIME_STEP_HOURS for t in 1:num_time_steps))

    optimize_and_check!(model)

    p_charge_ida_value = value.(p_charge_ida)
    p_discharge_ida_value = value.(p_discharge_ida)
    p_curtailed_ida_value = value.(p_curtailed_ida)
    p_close_charge_daa_value = value.(p_close_charge_daa)
    p_close_discharge_daa_value = value.(p_close_discharge_daa)
    p_close_curtailed_daa_value = value.(p_close_curtailed_daa)

    return (
        ida=ida,
        p_charge_ida=p_charge_ida_value,
        p_discharge_ida=p_discharge_ida_value,
        p_close_charge_daa=p_close_charge_daa_value,
        p_close_discharge_daa=p_close_discharge_daa_value,
        p_curtailed_ida=p_curtailed_ida_value,
        p_close_curtailed_daa=p_close_curtailed_daa_value,
        p_curtailed_daa_ida=p_curtailed_daa .- p_close_curtailed_daa_value .+ p_curtailed_ida_value,
        p_charge_daa_ida=value.(p_charge_daa_ida),
        p_discharge_daa_ida=value.(p_discharge_daa_ida),
        soc=value.(soc[1:num_time_steps]),
        p_inject_ida=value.(p_inject_ida),
    )
end

function get_idc_schedule(
    idc,
    market_premium,
    pv_output,
    p_limit,
    storage_capacity,
    p_charge_max,
    p_discharge_max,
    delta_p_perm,
    p_charge_daa_ida,
    p_discharge_daa_ida,
    p_curtailed_daa_ida,
    number_of_cycles::Integer=2,
    efficiency=0.95,
    start_soc=0.0,
    end_soc=0.0;
    optimizer=HiGHS.Optimizer,
    # optimizer=Cbc.Optimizer
)
    idc = to_float_vector(idc)
    market_premium = to_float_vector(market_premium)
    pv_output = to_float_vector(pv_output)
    p_charge_daa_ida = to_float_vector(p_charge_daa_ida)
    p_discharge_daa_ida = to_float_vector(p_discharge_daa_ida)
    p_curtailed_daa_ida = to_float_vector(p_curtailed_daa_ida)
    validate_lengths(
        idc,
        market_premium,
        pv_output,
        p_charge_daa_ida,
        p_discharge_daa_ida,
        p_curtailed_daa_ida;
        names=["market_premium", "pv_output", "p_charge_daa_ida", "p_discharge_daa_ida", "p_curtailed_daa_ida"],
    )

    num_time_steps = length(idc)
    model = Model(optimizer)

    @variable(model, p_charge_idc[1:num_time_steps] >= 0)
    @variable(model, p_discharge_idc[1:num_time_steps] >= 0)
    @variable(model, p_curtailed_idc[1:num_time_steps] >= 0)
    @variable(model, p_close_charge_daa_ida[1:num_time_steps] >= 0)
    @variable(model, p_close_discharge_daa_ida[1:num_time_steps] >= 0)
    @variable(model, p_close_curtailed_daa_ida[1:num_time_steps] >= 0)
    @variable(model, soc[1:num_time_steps + 1] >= 0)
    @variable(model, p_inject_idc[1:num_time_steps] >= 0)
    @variable(model, p_charge_daa_ida_idc[1:num_time_steps] >= 0)
    @variable(model, p_discharge_daa_ida_idc[1:num_time_steps] >= 0)
    @variable(model, z[1:num_time_steps], Bin)
    @variable(model, y[1:num_time_steps], Bin)
    @variable(model, x[1:num_time_steps], Bin)
    @variable(model, w[1:num_time_steps], Bin)
    @variable(model, v[1:num_time_steps], Bin)

    @constraint(model, [t in 1:num_time_steps],
        soc[t + 1] == soc[t] + p_charge_idc[t] * DEFAULT_TIME_STEP_HOURS * efficiency - p_discharge_idc[t] * DEFAULT_TIME_STEP_HOURS / efficiency - p_close_charge_daa_ida[t] * DEFAULT_TIME_STEP_HOURS * efficiency + p_close_discharge_daa_ida[t] * DEFAULT_TIME_STEP_HOURS / efficiency + p_charge_daa_ida[t] * DEFAULT_TIME_STEP_HOURS * efficiency - p_discharge_daa_ida[t] * DEFAULT_TIME_STEP_HOURS / efficiency)
    @constraint(model, [t in 1:num_time_steps + 1], soc[t] <= storage_capacity)
    @constraint(model,
        sum((p_charge_daa_ida[t] + p_charge_idc[t] - p_close_charge_daa_ida[t]) * DEFAULT_TIME_STEP_HOURS for t in 1:num_time_steps) <= number_of_cycles * storage_capacity)
    @constraint(model, soc[1] == start_soc)
    @constraint(model, soc[num_time_steps + 1] == end_soc)

    @constraint(model, [t in 1:num_time_steps],
        p_discharge_idc[t] - p_charge_idc[t] - p_curtailed_idc[t] + p_discharge_daa_ida[t] - p_charge_daa_ida[t] + p_close_charge_daa_ida[t] - p_close_discharge_daa_ida[t] + p_close_curtailed_daa_ida[t] + pv_output[t] - p_curtailed_daa_ida[t] >= 0)
    @constraint(model, [t in 1:num_time_steps],
        p_discharge_idc[t] - p_charge_idc[t] - p_curtailed_idc[t] + p_discharge_daa_ida[t] - p_charge_daa_ida[t] + p_close_charge_daa_ida[t] - p_close_discharge_daa_ida[t] + p_close_curtailed_daa_ida[t] + pv_output[t] - p_curtailed_daa_ida[t] <= p_limit)

    @constraint(model, [t in 1:num_time_steps], p_charge_idc[t] - p_close_charge_daa_ida[t] + p_charge_daa_ida[t] <= p_charge_max)
    @constraint(model, [t in 1:num_time_steps], p_charge_idc[t] - p_close_charge_daa_ida[t] + p_charge_daa_ida[t] >= 0)
    @constraint(model, [t in 1:num_time_steps], p_discharge_idc[t] - p_close_discharge_daa_ida[t] + p_discharge_daa_ida[t] <= p_discharge_max)
    @constraint(model, [t in 1:num_time_steps], p_discharge_idc[t] - p_close_discharge_daa_ida[t] + p_discharge_daa_ida[t] >= 0)
    @constraint(model, [t in 1:num_time_steps], p_curtailed_idc[t] <= pv_output[t] - p_curtailed_daa_ida[t] + p_close_curtailed_daa_ida[t])

    @constraint(model, [t in 1:num_time_steps], p_close_curtailed_daa_ida[t] <= p_curtailed_daa_ida[t] * (1 - w[t]))
    @constraint(model, [t in 1:num_time_steps], p_close_discharge_daa_ida[t] <= p_discharge_daa_ida[t] * (1 - y[t]))
    @constraint(model, [t in 1:num_time_steps], p_close_charge_daa_ida[t] <= p_charge_daa_ida[t] * (1 - x[t]))

    @constraint(model, [t in 1:num_time_steps], p_charge_idc[t] <= p_charge_max * z[t])
    @constraint(model, [t in 1:num_time_steps], p_discharge_idc[t] <= p_discharge_max * (1 - z[t]))
    @constraint(model, [t in 1:num_time_steps], p_charge_idc[t] <= p_charge_max * x[t])
    @constraint(model, [t in 1:num_time_steps], p_discharge_idc[t] <= p_discharge_max * y[t])
    @constraint(model, [t in 1:num_time_steps], p_curtailed_idc[t] <= (pv_output[t] - p_curtailed_daa_ida[t]) * w[t])
    @constraint(model, [t in 1:num_time_steps], p_curtailed_idc[t] <= pv_output[t] * (1 - v[t]))
    @constraint(model, [t in 1:num_time_steps], p_discharge_idc[t] <= p_discharge_max * v[t])

    @constraint(model, [t in 1:num_time_steps],
        p_inject_idc[t] == p_discharge_idc[t] - p_charge_idc[t] + p_discharge_daa_ida[t] - p_charge_daa_ida[t] + p_close_charge_daa_ida[t] - p_close_discharge_daa_ida[t] + pv_output[t] - p_curtailed_daa_ida[t] + p_close_curtailed_daa_ida[t] - p_curtailed_idc[t])
    @constraint(model, [t in 1:num_time_steps], p_charge_idc[t] - p_close_charge_daa_ida[t] + p_charge_daa_ida[t] == p_charge_daa_ida_idc[t])
    @constraint(model, [t in 1:num_time_steps], p_discharge_idc[t] - p_close_discharge_daa_ida[t] + p_discharge_daa_ida[t] == p_discharge_daa_ida_idc[t])

    @constraint(model, [t in 2:num_time_steps],
        (p_charge_daa_ida_idc[t] - p_discharge_daa_ida_idc[t]) - (p_charge_daa_ida_idc[t - 1] - p_discharge_daa_ida_idc[t - 1]) <= delta_p_perm)
    @constraint(model, [t in 2:num_time_steps],
        (p_charge_daa_ida_idc[t - 1] - p_discharge_daa_ida_idc[t - 1]) - (p_charge_daa_ida_idc[t] - p_discharge_daa_ida_idc[t]) <= delta_p_perm)

    @objective(model, Max,
        sum(idc[t] * (p_discharge_idc[t] - p_charge_idc[t] + p_close_charge_daa_ida[t] - p_close_discharge_daa_ida[t] + p_close_curtailed_daa_ida[t] - p_curtailed_idc[t]) * DEFAULT_TIME_STEP_HOURS + market_premium[t] * p_inject_idc[t] * DEFAULT_TIME_STEP_HOURS for t in 1:num_time_steps))

    optimize_and_check!(model)

    p_charge_idc_value = value.(p_charge_idc)
    p_discharge_idc_value = value.(p_discharge_idc)
    p_curtailed_idc_value = value.(p_curtailed_idc)
    p_close_charge_daa_ida_value = value.(p_close_charge_daa_ida)
    p_close_discharge_daa_ida_value = value.(p_close_discharge_daa_ida)
    p_close_curtailed_daa_ida_value = value.(p_close_curtailed_daa_ida)

    return (
        idc=idc,
        p_charge_idc=p_charge_idc_value,
        p_discharge_idc=p_discharge_idc_value,
        p_close_charge_daa_ida=p_close_charge_daa_ida_value,
        p_close_discharge_daa_ida=p_close_discharge_daa_ida_value,
        p_curtailed_idc=p_curtailed_idc_value,
        p_close_curtailed_daa_ida=p_close_curtailed_daa_ida_value,
        p_curtailed_daa_ida_idc=p_curtailed_daa_ida .- p_close_curtailed_daa_ida_value .+ p_curtailed_idc_value,
        p_charge_daa_ida_idc=value.(p_charge_daa_ida_idc),
        p_discharge_daa_ida_idc=value.(p_discharge_daa_ida_idc),
        soc=value.(soc[1:num_time_steps]),
        p_inject_idc=value.(p_inject_idc),
    )
end

end