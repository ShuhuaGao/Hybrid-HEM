# Mixed-integer linear programming (MILP) as the expert in IL

"""
    run_MILP!(s::DataFrame, h::Dict) -> JuMP.Model

Assuming all future information is known in the scenario `s`, we run MILP to solve the optimization 
problem and get the ideally optimal solutions. The solutions are written into `s` in place, while the
modified `s` is also returned for convenience. The load specifications of the home are presented by `h`.

The JuMP model after optimization is returned. 
"""
function run_MILP!(s::DataFrame, h::Dict)::JuMP.Model
    ρ = s.ρ
    PPV = s.PPV

    model = build_model(h, PPV, ρ)
    optimize!(model)
    if termination_status(model) == OPTIMAL
        # @show objective_value(model)
    else
        @warn "Not OPTIMAL"
    end
    write_solution!(s, model, h)
    return model
end


function add_CR_load!(model::JuMP.Model, load::Dict, ρ::AbstractVector)
    # a CR load does not affect optimization
    # we simply store some constants here for convenience
    id = load["id"]
    P = load["P"]::Real
    model[Symbol("P_", id)] = fill(float(P), model[:T])
    model[Symbol("C_", id)] = ρ .* float(P)
    return model
end

"""
    add_AD_load!(model::JuMP.Model, load::Dict, ρ::AbstractVector; K::Integer=10) -> JuMP.Model

Add an adjustable `load` to `model`, whose quadratic cost is approximated by `K` piece linearization.
"""
function add_AD_load!(model::JuMP.Model, load::Dict, ρ::AbstractVector; K::Integer = 10)
    @assert K >= 3
    T = model[:T]
    @assert length(ρ) == T
    Pmin, Pmax = load["Pmin"], load["Pmax"]
    id = load["id"]
    model[Symbol("P_", id)] = P = @variable(model, [1:T], lower_bound = Pmin, upper_bound = Pmax)
    model[Symbol("C_", id)] = C = @variable(model, [1:T])
    # PLF approximation of the quadratic cost
    # note that we cannot separate the electricity cost and the discomfort cost here 
    for t = 1:T
        f(P) = ρ[t] * P + load["α"] * (P - Pmax)^2
        xs, ks, bs = linearize(f, Pmin, Pmax, K)
        @constraint(model, C[t] .>= ks .* P[t] .+ bs)
    end
    # store discomfort cost for more info
    model[Symbol("C_", id, "_dc")] = @expression(model, load["α"] .* (P .- Pmax) .^ 2)
    return model
end

"""
    add_SU_load!(model::JuMP.Model, load::Dict, ρ::AbstractVector)

Add an SU `load` to `model` with the price profile `ρ`.
"""
function add_SU_load!(model::JuMP.Model, load::Dict, ρ::AbstractVector)
    T = model[:T]
    L = load["L"]
    ts, tf = load["ts"], load["tf"]
    id = load["id"]
    model[Symbol("z_", id)] = z = @variable(model, [2-L:T], Bin)    # DenseAxisArray
    @constraint(model, z[2-L:ts-1] .== 0)
    @constraint(model, z[tf-L+2:T] .== 0)
    # @constraint(model, sum(z) == 1)  # equiv. to the above two lines
    @constraint(model, sum(z[ts:tf-L+1]) == 1)
    # define P as affine expressions since z is the control variable
    model[Symbol("P_", id)] = P = @expression(model, zeros(AffExpr, T))
    Prate = load["P"]
    @assert Prate > 0
    for t = 1:T
        P[t] = @expression(model, Prate * sum(z[t-L+1:t]))
    end
    model[Symbol("C_", id)] = @expression(model, ρ .* P)
    return model
end

"""
    add_SI_load!(model::JuMP.Model, load::Dict, ρ::AbstractVector)

Add an SI `load` to `model` with the price profile `ρ`.
"""
function add_SI_load!(model::JuMP.Model, load::Dict, ρ::AbstractVector)
    T = model[:T]
    L = load["L"]
    ts, tf = load["ts"], load["tf"]
    id = load["id"]
    Prate = load["P"]
    # only need to define binary variables between ts and tf
    model[Symbol("z_", id)] = z = @variable(model, [1:T], Bin)
    @constraint(model, z[1:ts-1] .== 0)
    @constraint(model, z[tf+1:end] .== 0)
    @constraint(model, sum(z[ts:tf]) == L)
    model[Symbol("P_", id)] = P = @expression(model, Prate .* z)
    model[Symbol("C_", id)] = @expression(model, ρ .* P)
    return model
end

"""
    setup_power_exchange!(model::JuMP.Model, g::Dict, ρ::AbstractVector)

Setup the power exchange with the utility grid configured by `g`. The electricity purchase price
profile is given by `ρ`. 
"""
function setup_power_exchange!(model::JuMP.Model, g::Dict, ρ::AbstractVector)
    Pmax = g["Pmax"] # any large value is OK
    β = g["β"]
    @assert 0 <= β <= 1
    T = model[:T]
    @variable(model, zb[1:T], Bin)
    @variable(model, zs[1:T], Bin)
    @variable(model, Pb[1:T] >= 0)
    @variable(model, Ps[1:T] >= 0)
    @constraint(model, Pb .<= zb .* Pmax)
    @constraint(model, Ps .<= zs .* Pmax)
    @constraint(model, zb .+ zs .<= 1)
    ρs = β .* ρ
    @expression(model, Cele, ρ .* Pb - ρs .* Ps)
    return model
end


"""
    build_model(h::Dict, PPV::AbstractVector, ρ::AbstractVector) -> JuMP.Model

Given the PV generation `PPV` and price `ρ` of `T` steps, build an MILP model for HEMS with home
configuration `h`.
"""
function build_model(h::Dict, PPV::AbstractVector, ρ::AbstractVector)::JuMP.Model
    T = length(PPV)
    @assert T == length(ρ)
    model = JuMP.Model(Cbc.Optimizer)
    set_optimizer_attribute(model, "logLevel", 0)
    model[:T] = T
    # add each load (no need to involve the critical load)
    for l in h["CR"]
        add_CR_load!(model, l, ρ)
    end
    for l in h["AD"]
        add_AD_load!(model, l, ρ; K = 20)
    end
    for l in h["SU"]
        add_SU_load!(model, l, ρ)
    end
    for l in h["SI"]
        add_SI_load!(model, l, ρ)
    end
    # add the utility grid 
    g = h["UtilityGrid"]
    setup_power_exchange!(model, g, ρ)
    # total load consumption
    @expression(model, P_load, sum(model[Symbol("P_", l["id"])]
                                   for l in Iterators.flatten((h["CR"], h["AD"], h["SU"], h["SI"]))))
    @expression(model, C_load, sum(model[Symbol("C_", l["id"])]
                                   for l in Iterators.flatten((h["CR"], h["AD"], h["SU"], h["SI"]))))
    # set the power balance
    @constraint(model, cst_power_balance, P_load .- PPV .== model[:Pb] .- model[:Ps])
    # set up objective 
    ρs = g["β"] .* ρ
    @expression(model, C, (C_load .- ρ .* PPV .+ (ρ .- ρs) .* model[:Ps]))
    @objective(model, Min, sum(C))
    return model
end

"""
    write_solution!(sc::DataFrame, model::JuMP.Model, h::Dict; CR::Bool=false) -> DataFrame

Write the optimal solution of related variables in the optimized `model` into the scenario dataframe
`sc` in place. The updated dataframe is also returned. `h` refers to the home configuration.

The keyword argument `CR` controls whether the constant power and corresponding cost of a critical
load is written into `sc` (default = `false`). 
"""
function write_solution!(sc::DataFrame, model::JuMP.Model, h::Dict; CR::Bool = false)
    if termination_status(model) != OPTIMAL
        @warn "The termination status of the model is '$(termination_status(model))'"
    end
    T = model[:T]
    if CR
        for l in h["CR"]
            id = l["id"]
            for pre in ["P_", "C_"]
                key = Symbol(pre, id)
                sc[!, key] = value.(model[key])
            end
        end
    end
    for l in h["AD"]
        id = l["id"]
        for pre in ["P_", "C_"]
            key = Symbol(pre, id)
            sc[!, key] = value.(model[key])
        end
        dc_key = Symbol("C_", id, "_dc")
        sc[!, dc_key] = value.(model[dc_key])
    end
    for l in h["SU"]
        id = l["id"]
        # handle z_ specially because its container is a DenseAxisArray (instead of a Vector)
        zkey = Symbol("z_", id)
        sc[!, zkey] = [value(model[zkey][i]) for i = 1:T]
        for pre in ["P_", "C_"]
            key = Symbol(pre, id)
            sc[!, key] = value.(model[key])[1:T]    # z_ starts from a negative index
        end
    end
    for l in h["SI"]
        id = l["id"]
        for pre in ["z_", "P_", "C_"]
            key = Symbol(pre, id)
            sc[!, key] = value.(model[key])    # z_ starts from a negative index
        end
    end
    for key in (:zb, :zs, :Pb, :Ps, :Cele)
        sc[!, key] = value.(model[key])
    end
    for key in (:P_load, :C_load, :C)
        sc[!, key] = value.(model[key])
    end
    return sc
end