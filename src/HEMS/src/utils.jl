# Read TOML file 

"""
    read_home_config(toml_file::String) -> Dict

Read home configuration from a TOML file.
"""
function read_home_config(toml_file::String)::Dict
    h = TOML.parsefile(toml_file)
    # confirm unique ID
    reserved = Set(["load"])
    ids = Set{String}()
    for id in get_all_load_ids(h)
        if id in ids
            error("The ID '$id' is not unique!")
        end
        if id in reserved
            error("The ID '$id' is reserved and should not be used!")
        end
        push!(ids, id)
    end
    return h
end

"""
    get_all_load_ids(h::Dict) -> Vector{String}

Get the ID of all loads in the home `h`.
"""
function get_all_load_ids(h::Dict)::Vector{String}
    return [l["id"] for l in Iterators.flatten((h["CR"], h["AD"], h["SI"], h["SU"]))]
end


function validate_solution(s::AbstractDataFrame, h::AbstractDict)
    C_SI = C_SU = C_AD = C_CR = 0.0
    # shiftable constraints
    for l in h["SU"]
        id = l["id"]
        if !isapprox(sum(s[!, Symbol("z_", id)]), 1)    # z_ is stored as Float in `s`
            @show sum(s[!, Symbol("z_", id)])
            error("Infeasible solution in SU load $id")
        end
        C_SU += sum(s[!, Symbol("C_", id)])
    end
    for l in h["SI"]
        id = l["id"]
        if !isapprox(sum(s[!, Symbol("z_", id)]), l["L"])
            error("Infeasible solution in SI load $id")
        end
        C_SI += sum(s[!, Symbol("C_", id)])
    end
    # power balance
    consumption = sum(s[!, Symbol("P_", l["id"])] for l in Iterators.Flatten((h["AD"], h["SI"], h["SU"]))) +
                  s.Ps .+ sum(l["P"] for l in h["CR"])
    generation = s.Pb .+ s.PPV
    if !all(consumption .≈ generation)
        @show consumption generation
        error("Power not balanced!")
    end
    for l in h["AD"]
        C_AD += sum(s[!, Symbol("C_", l["id"])])
    end
    for l in h["CR"]
        C_CR += sum(l["P"] .* s.ρ)
    end
    ρ = s.ρ
    ρs = h["UtilityGrid"]["β"] .* ρ
    return C_SI + C_SU + C_AD + C_CR - sum(ρ .* s.PPV) + sum((ρ .- ρs) .* s.Ps)
end

"""
    compute_cost!(df::DataFrame, h::AbstractDict)->Float64

For home `h`, `df` contains price, PV, and the power of each load. Then, the net power exchange is 
computed and stored into `df` as `Pb` and `Ps` columns. Finally, the stepwise electricity cost `Cele` is 
calculated, and the total cost `C` is also computed, both stored into `df` in place.

The total cost of all steps is returned.
"""
function compute_cost!(df::DataFrame, h::AbstractDict)::Float64
    T = nrow(df)
    c = 0.0
    for l in h["CR"]
        df[!, "P_$(l["id"])"] = fill(l["P"], T)
    end
    ids = [l["id"] for l in Iterators.Flatten((h["CR"], h["AD"], h["SU"], h["SI"]))]
    Pnet = sum(df[!, "P_$id"] for id in ids) .- df.PPV
    df.Pnet = Pnet
    Pb = max.(Pnet, 0.0)
    Ps = -min.(Pnet, 0.0)
    ρb = df.ρ
    ρs = h["UtilityGrid"]["β"] .* ρb
    Cele = @. ρb * Pb - ρs * Ps
    df.Pb = Pb
    df.Ps = Ps
    df.Cele = Cele
    # discomfort cost
    Cdc = zeros(T)
    for l in h["AD"]
        id = l["id"]
        Pmax = l["Pmax"]
        α = l["α"]
        df[!, "C_$(id)_dc"] = @. α * (Pmax - df[!, "P_$id"])^2
        Cdc .+= df[!, "C_$(id)_dc"]
    end
    df.Cdc = Cdc
    df.C = df.Cdc .+ df.Cele
    return sum(df.C)
end