module DynareOptions

export dynare_options

type dynare_options
    dynare_version::String
    function dynare_options()
        dynare_version = ""
        new(dynare_version)
    end
end

end
