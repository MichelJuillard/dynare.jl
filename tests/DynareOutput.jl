module DynareOutput

export dynare_output

type dynare_output
    dynare_version::String
    steady_state::Array{Float64,1}
    exo_steady_state::Array{Float64,1}
    function dynare_output()
        dynare_version = ""
        steady_state = Array(Float64,0)
        exo_steady_state = Array(Float64,0)
        new(dynare_version,steady_state,exo_steady_state)
    end
end

end
