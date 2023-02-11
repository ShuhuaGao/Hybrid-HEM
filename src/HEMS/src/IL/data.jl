# Prepare data for IL 
# 1. run MILP to get the optimal solutions for each scenario
# 2. extract state-action dataset of each load for DNN training in behavior cloning
# In the returned dataset (X, y), each column of X is a state, and each value of y is the
# corresponding scalar action.

"""
    run_expert!(ss::AbstractVector{DataFrame}, h::Dict; 
        out_bson::Union{String, Nothing} = nothing) -> Vector{DataFrame}

Run an IL expert to get optimal demonstrations for each scenario in `ss`. The optimal solutions are 
written into `ss` in place, while the updated `ss` is also returned.

Besides, the updated `ss` is stored into the BSON file `out_bson` with key `:ss`.
"""
function run_expert!(ss::AbstractVector{DataFrame}, h::Dict;
    out_bson::Union{String,Nothing} = nothing, report_progress=false)
    train_ss = DataFrame[]
    for (i, s) in enumerate(ss)
        # the solutions have been injected into `s` in place
        if report_progress
            println("Processing $i/$(length(ss))")
            flush(stdout)
        end
        run_MILP!(s, h)
        push!(train_ss, s)
    end
    if out_bson !== nothing
        @save out_bson ss = train_ss
    end
    return train_ss
end


"""
    build_expert_dataset(ss::AbstractVector{DataFrame}, h::Dict; 
        out_bson::Union{String,Nothing} = nothing) -> Dict

Build expert demonstrations dataset for each load in `h` based on optimized scenarios `ss`. 
The dataset (X, y) of a load with ID `id` can be retrieved with key `id` from the returned `Dict`.

Each column of the returned X denotes an input example (i.e., a state).
The returned y is a vector.

If the file path `out_bson` is specified, then the returned `Dict` is also stored into the BSON 
file with key `:d`.
"""
function build_expert_dataset(ss::AbstractVector{DataFrame}, h::Dict; H=1, full::Bool=false,
    out_bson::Union{String,Nothing} = nothing)::Dict
    d = Dict(l["id"] => build_AD_dataset(ss, l) for l in h["AD"])
    if full
        for l in h["SU"]
            d[l["id"]] = build_full_SU_dataset(ss, l)
        end
        for l in h["SI"]
            d[l["id"]] = build_full_SI_dataset(ss, l)
        end
    else
        for l in h["SU"]
            d[l["id"]] = build_SU_dataset(ss, l; H)
        end
        for l in h["SI"]
            d[l["id"]] = build_SI_dataset(ss, l)
        end
    end
    if !isnothing(out_bson)
        @save out_bson d
    end
    return d
end


"""
    build_AD_dataset(ss::AbstractVector{DataFrame}, id::String) -> (Matrix, Vector)

Create a dataset for an AD load with ID `id` from scenarios `ss`.
State: ``[t, ρ, P^{PV}]``. Action: ``P^{AD}``.
"""
function build_AD_dataset(ss::AbstractVector{DataFrame}, id::String)
    n = length(ss)
    T = nrow(ss[1])
    X = zeros(3, n * T)
    y = zeros(n * T)
    i = 1
    for s in ss
        for t = 1:T
            X[:, i] .= [t, s.ρ[t], s.PPV[t]]
            y[i] = s[t, "P_$id"]
            i += 1
        end
    end
    @assert i == length(y) + 1
    return X, y
end

"""
    build_AD_dataset(ss::AbstractVector{DataFrame}, l::Dict) -> (Matrix, Vector)

Create a dataset for an AD load `l`. 
See [`build_AD_dataset(ss::AbstractVector{DataFrame}, id::String)`](@ref).
"""
build_AD_dataset(ss::AbstractVector{DataFrame}, l::Dict) = build_AD_dataset(ss, l["id"])


"""
    build_SU_dataset(ss::AbstractVector{DataFrame}, l::Dict) -> (Matrix, Vector)

Create a dataset for a SU load `l` from scenarios `ss`.
State: ``[t, ρ, P^{PV}]``. Action: ``z^{su}``.
Note that we only consider ``t \\in [ts, tf-L+1]``, in which the action may be 1.
"""
function build_SU_dataset(ss::AbstractVector{DataFrame}, l::Dict, ls_AD=nothing; H::Integer = 1)
    @assert H >= 1
    n = length(ss)
    ts, tf, L = l["ts"], l["tf"], l["L"]
    m = tf - L + 1 - ts + 1 # number of state-action pairs in each scenario 
    n_AD = 0
    if ls_AD !== nothing
        n_AD = length(ls_AD)
    end
    X = zeros(1 + 2 * H + n_AD, n * m)
    y = zeros(n * m)
    i = 1
    id = l["id"]
    for s in ss
        count_one = 0    # each scenario has exactly one 1
        for t = ts:tf-L+1
            # state: t and historical price & PV generation
            X[1, i] = t
            X[2:2+H-1, i] .= (s.ρ[h] for h = t-H+1:t)
            X[2+H:2+H+H-1, i] .= (s.PPV[h] for h = t-H+1:t)
            # binary variables may not be exactly 0 or 1 in JuMP
            if s[t, "z_$id"] > 0.5
                y[i] = 1
                count_one += 1
            else
                y[i] = 0
            end
            if ls_AD !== nothing
                X[2+2*H:end, i] .= (s[t, "P_$(lad["id"])"] for lad in ls_AD)
            end
            i += 1
        end
        @assert count_one == 1
    end
    @assert i == length(y) + 1
    return X, y
end


"""
    build_SI_dataset(ss::AbstractVector{DataFrame}, l::Dict) -> (Matrix, Vector)

Create a dataset for a SI load `l` from scenarios `ss`.
State: ``[t, ρ, P^{PV}, O^{si}]``. Action: ``z^{si}``.
Note that we only consider ``t \\in [ts, tf]``, in which the action may be 1.
"""
function build_SI_dataset(ss::AbstractVector{DataFrame}, l::Dict)
    n = length(ss)
    ts, tf, L = l["ts"], l["tf"], l["L"]
    m = tf - ts + 1 # number of state-action pairs in each scenario 
    X = zeros(4, n * m)
    y = zeros(n * m)
    i = 1
    id = l["id"]
    i_s = 0
    for s in ss
        i_s += 1
        Osi = 0
        for t = ts:tf
            X[:, i] .= [t, s.ρ[t], s.PPV[t], Osi]
            y[i] = s[t, "z_$id"] > 0.5 ? 1 : 0 # JuMP may not yield exactly 1 though it is a binary
            if y[i] == 1  
                Osi += 1
            end
            i += 1
        end
        @assert Osi == L
    end
    @assert i == length(y) + 1
    return X, y
end

"""
    build_full_SU_dataset(ss::AbstractVector{DataFrame}, l::Dict) -> (Matrix, Vector)

Build a *full* state-action dataset for a SU load `l`, which contains the state and action pair at 
each time step, even if the time is outside the allowed range of `l`.

`(X, y)` is returned, where each column of X and the corresponding item in y forms a state-action pair.
"""
function build_full_SU_dataset(ss::AbstractVector{DataFrame}, l::Dict)
    n = length(ss)
    T = nrow(ss[1])
    ts, tf, L = l["ts"], l["tf"], l["L"]
    id = l["id"]
    X = fill(NaN, (3, n * T))
    y = fill(NaN, n * T)
    i = 1
    for s in ss
        for t = 1:T
            X[:, i] .= [t, s.ρ[t], s.PPV[t]]
            y[i] = s[t, "z_$id"] > 0.5 ? 1 : 0
            i += 1
        end
    end
    @assert i == length(y) + 1
    @assert all(isfinite, X)
    @assert all(isfinite, y)
    return X, y
end


"""
    build_full_SI_dataset(ss::AbstractVector{DataFrame}, l::Dict) -> (Matrix, Vector)

Build a *full* state-action dataset for a SI load `l`, which contains the state and action pair at 
each time step, even if the time is outside the allowed range of `l`.

`(X, y)` is returned, where each column of X and the corresponding item in y forms a state-action pair.
"""
function build_full_SI_dataset(ss::AbstractVector{DataFrame}, l::Dict)
    n = length(ss)
    T = nrow(ss[1])
    ts, tf, L = l["ts"], l["tf"], l["L"]
    id = l["id"]
    X = fill(NaN, (4, n * T))
    y = fill(NaN, n * T)
    i = 1
    for s in ss
        Osi = 0  # counter O
        for t = 1:T
            X[:, i] .= [t, s.ρ[t], s.PPV[t], Osi]
            y[i] = s[t, "z_$id"] > 0.5 ? 1 : 0
            if y[i] == 1
                Osi += 1
            end
            i += 1
        end
    end
    @assert i == length(y) + 1
    @assert all(isfinite, X)
    @assert all(isfinite, y)
    return X, y
end