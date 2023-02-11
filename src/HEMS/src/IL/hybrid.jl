# Optimize AD loads after the shiftable load's decisions


# function optimize_AD_loads(h::AbstractDict, ρ::Float64, PPV::Float64,
#     z_shiftable::AbstractDict; K::Integer = 20)
#     @assert length(h["SU"]) + length(h["SI"]) == length(z_shiftable)
#     id_to_P = Dict()
#     P_shiftable = Float64[]
#     for (id, z) in z_shiftable
#         push!(P_shiftable, z * )
#     end
# end

function optimize_AD_loads(h::AbstractDict, ρ::AbstractFloat, PPV::AbstractFloat,
    P_shiftable::AbstractVector; K::Integer = 20, verbose=false)
    ls = h["AD"]
    g = h["UtilityGrid"]
    @assert length(h["SU"]) + length(h["SI"]) == length(P_shiftable)
    model = JuMP.Model(Cbc.Optimizer)
    set_optimizer_attribute(model, "logLevel", 0)
    # only consider the current time t
    for load in ls
        add_AD_load_t!(model, load, ρ; K)
    end
    setup_power_exchange_t!(model, g, ρ)
    # power balance
    P_other = sum(P_shiftable) + sum(l["P"] for l in h["CR"])   # critical and shiftable load consumption
    @constraint(model, cst_power_balance, P_other + sum(model[Symbol("P_", l["id"])]
                                                        for l in ls) == model[:Pb] - model[:Ps] + PPV)
    ρs = g["β"] .* ρ
    @objective(model, Min, sum(model[Symbol("C_", l["id"])] for l in ls) + (ρ - ρs) * model[:Ps])

    optimize!(model)
    if termination_status(model) == OPTIMAL
        # compute total cost
        # @show objective_value(model)
    else
        @warn "Not OPTIMAL"
    end
    if verbose
        @show P_other value(sum(model[Symbol("P_", l["id"])] for l in ls)) value(model[:Pb]) value(model[:Ps]) PPV
    end
    return model
end


function setup_power_exchange_t!(model, g, ρ)
    Pmax = g["Pmax"] # any large value is OK
    β = g["β"]
    @assert 0 <= β <= 1
    @variable(model, zb, Bin)
    @variable(model, zs, Bin)
    @variable(model, 0 <= Pb <= Pmax)
    @variable(model, 0 <= Ps <= Pmax)
    @constraint(model, Pb <= zb * Pmax)
    @constraint(model, Ps <= zs * Pmax)
    @constraint(model, zb + zs <= 1)
    ρs = β .* ρ
    @expression(model, Cele, ρ .* Pb - ρs .* Ps)
    return model
end


function add_AD_load_t!(model::JuMP.Model, load::AbstractDict, ρ::Float64; K::Integer = 20)
    @assert K >= 3
    Pmin, Pmax = load["Pmin"], load["Pmax"]
    id = load["id"]
    model[Symbol("P_", id)] = P = @variable(model, lower_bound = Pmin, upper_bound = Pmax)
    model[Symbol("C_", id)] = C = @variable(model)
    f(P) = ρ * P + load["α"] * (P - Pmax)^2
    xs, ks, bs = linearize(f, Pmin, Pmax, K)
    @constraint(model, C .>= ks .* P .+ bs)
    model[Symbol("C_", id, "_dc")] = @expression(model, load["α"] .* (P .- Pmax) .^ 2)
    return model
end

"""
    manage_loads(s::DataFrame, h::AbstractDict, agents::AbstractDict; K::Integer = 20) -> DataFrame

Given the `agents` managing shiftable loads, perform load management for home `h` in scenario `s`. 
`K` is the number of segments in linear approximation of the quadratic cost.

The management strategy is returned in a new data frame along with the original scenario info.
"""
function manage_loads(s::DataFrame, h::AbstractDict, agents::AbstractDict; K::Integer = 20)
    s = s[:, [:ρ, :PPV]]    # a copy due to :
    ρ, PPV = s.ρ, s.PPV
    T = size(s)[1]
    # insert relevant columns
    for l in h["AD"]
        s[!, Symbol("P_", l["id"])] .= fill(NaN, T)
        s[!, Symbol("C_", l["id"])] .= fill(NaN, T)
        s[!, Symbol("C_", l["id"], "_dc")] .= fill(NaN, T)
    end
    for l in Iterators.Flatten((h["SU"], h["SI"]))
        s[!, Symbol("z_", l["id"])] .= fill(NaN, T)
        s[!, Symbol("P_", l["id"])] .= fill(NaN, T)
        s[!, Symbol("C_", l["id"])] .= fill(NaN, T)
    end
    for col in (:zb, :zs, :Pb, :Ps)
        s[!, col] = fill(NaN, T)
    end
    # schedule shiftable loads 
    for l in h["SU"]
        schedule_SU_load!(s, l, agents[l["id"]])
    end
    for l in h["SI"]
        schedule_SI_load!(s, l, agents[l["id"]])
    end
    # AD loads
    info_AD = Dict(l["id"] => (l["α"], l["Pmax"]) for l in h["AD"])
    # @time begin 
    for t = 1:T
        P_shiftable = [s[t, Symbol("P_", l["id"])] for l in Iterators.Flatten((h["SU"], h["SI"]))]
        model = optimize_AD_loads(h, ρ[t], PPV[t], P_shiftable; K, verbose = false)
        Ps = Dict(l["id"] => value(model[Symbol("P_", l["id"])]) for l in h["AD"])
        for (id, P) in Ps
            s[t, Symbol("P_", id)] = P
            α, Pmax = info_AD[id]
            s[t, Symbol("C_", id)] = ρ[t] * P + α * (P - Pmax)^2
            s[t, Symbol("C_", id, "_dc")] = α * (P - Pmax)^2
        end
        # write also the grid power exchange
        for col in (:zb, :zs, :Pb, :Ps)
            s[t, col] = value(model[col])
        end
    end
    # end
    β = h["UtilityGrid"]["β"]
    s.Cele = ρ .* s.Pb .- β * ρ .* s.Ps
    #step-wise cost
    s.C = push!([s[!, Symbol("C_", l["id"], "_dc")] for l in h["AD"]], s.Cele) |> sum 
    return s
end

function schedule_SU_load!(s::DataFrame, l, agent)
    id, ts, tf = l["id"], l["ts"], l["tf"]
    P, L = l["P"], l["L"]
    s[!, Symbol("z_", id)] .= 0
    s[!, Symbol("P_", id)] .= 0.0
    s[!, Symbol("C_", id)] .= 0.0

    function switch_on(t)
        s[t, Symbol("z_", id)] = 1
        for dt = 0:L-1
            s[t+dt, Symbol("P_", id)] = P
            s[t+dt, Symbol("C_", id)] = s.ρ[t+dt] * P
        end
    end
    # action can be 1 only inside this time range
    for t = ts:tf-L+1
        state = [t, s.ρ[t], s.PPV[t]]
        action = only(agent(state)) >= 0.5
        if action == 1 || t == tf - L + 1  # tf - L + 1 is the last chance
            switch_on(t)
            break  # only a single 1 is allowed
        end
    end
    @assert sum(s[!, Symbol("z_", id)]) == 1
end

function schedule_SI_load!(s::DataFrame, l, agent)
    id, ts, tf = l["id"], l["ts"], l["tf"]
    P, L = l["P"], l["L"]
    s[!, Symbol("z_", id)] .= 0
    s[!, Symbol("P_", id)] .= 0.0
    s[!, Symbol("C_", id)] .= 0.0

    function keep_on(t)
        s[t, Symbol("z_", id)] = 1
        s[t, Symbol("P_", id)] = P
        s[t, Symbol("C_", id)] = s.ρ[t] * P
    end

    n_already_on = 0
    for t = ts:tf
        if tf - t + 1 == L - n_already_on   # all remaining steps must be ON
            for tr = t:tf
                keep_on(tr)
            end
            break
        end
        state = [t, s.ρ[t], s.PPV[t], n_already_on]
        action = only(agent(state)) >= 0.5
        if action == 1
            n_already_on += 1
            keep_on(t)
            if n_already_on == L
                break
            end
        end
    end
    @assert sum(s[!, Symbol("z_", id)]) == L
end