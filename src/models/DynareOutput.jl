module DynareOutput

export dynare_output

mutable struct dynare_output
    dynare_version::String
    steady_state::Vector{Float64}
    exo_steady_state::Vector{Float64}
    function dynare_output()
        dynare_version = ""
        steady_state = Vector{Float64}()
        exo_steady_state = Vector{Float64}()
        new(dynare_version,steady_state,exo_steady_state)
    end
end

end
