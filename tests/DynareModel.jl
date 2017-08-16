module DynareModel

export dynare_model

immutable Endo
    name::String
    long_name::String
    tex_name::String
    function Endo(name_arg,long_name_arg,tex_name_arg)
        name = name_arg
        long_name = long_name_arg
        tex_name = tex_name_arg
        new(name, long_name, tex_name)
    end
end

immutable Exo
    name::String
    long_name::String
    tex_name::String
    function Exo(name_arg,long_name_arg,tex_name_arg)
        name = name_arg
        long_name = long_name_arg
        tex_name = tex_name_arg
        new(name, long_name, tex_name)
    end
end

immutable Param
    name::String
    long_name::String
    tex_name::String
    function Param(name_arg,long_name_arg,tex_name_arg)
        name = name_arg
        long_name = long_name_arg
        tex_name = tex_name_arg
        new(name, long_name, tex_name)
    end
end

type dynare_model
    fname::String
    dynare_version::String
    sigma_e::Matrix{Float64}
    correlation_matrix::Matrix{Float64}
    orig_eq_nbr::Int64
    eq_nbr::Int64
    ramsey_eq_nbr::Int64
    h::Matrix{Float64}
    correlation_matrix_me::Matrix{Float64}
    endo::Vector{Endo}
    exo::Vector{Exo}
    param::Vector{Param}
    orig_endo_nbr::Int64
    lead_lag_incidence::Matrix{Int64}
    nstatic::Int64
    nfwrd::Int64
    npred::Int64
    nboth::Int64
    nsfwrd::Int64
    nspred::Int64
    ndynamic::Int64
    equation_tags::Vector{String}
    static_and_dynamic_models_differ::Bool
    exo_names_orig_ord::Vector{Int64}
    maximum_lag::Int64
    maximum_lead::Int64
    maximum_endo_lag::Int64
    maximum_endo_lead::Int64
    maximum_exo_lag::Int64
    maximum_exo_lead::Int64
    params::Vector{Float64}
    nnzderivatives::Vector{Int64}
    static::Function
    dynamic::Function
    user_written_analytical_steady_state::Bool
    steady_state::Function
    analytical_steady_state::Bool
    static_params_derivs::Function
    dynamic_params_derivs::Function

    function dynare_model()
        fname = ""
        dynare_version = ""
        sigma_e = Matrix{Float64}(0,0)
        correlation_matrix = Matrix{Float64}(0,0)
        orig_eq_nbr = 0
        eq_nbr = 0
        ramsey_eq_nbr = 0
        h = Matrix{Float64}(0,0)
        correlation_matrix_me = Matrix{Float64}(0,0)
        endo = []
        exo = []
        param = []
        orig_endo_nbr = 0
        lead_lag_incidence = Matrix{Int64}(0,0)
        nstatic = 0
        nfwrd = 0
        npred = 0
        nboth = 0
        nsfwrd = 0
        nspred = 0
        ndynamic = 0
        equation_tags = []
        static_and_dynamic_models_differ = false
        exo_names_orig_ord = []
        maximum_lag = 0
        maximum_lead = 0
        maximum_endo_lag = 0
        maximum_endo_lead = 0
        maximum_exo_lag = 0
        maximum_exo_lead = 0
        params = []
        nnzderivatives = []
        f_nothing() = nothing
        static = f_nothing
        dynamic = f_nothing
        user_written_analytical_steady_state = false
        steady_state = f_nothing
        analytical_steady_state = false
        static_params_derivs = f_nothing
        dynamic_params_derivs = f_nothing
        new()
#        new(fname, dynare_version, sigma_e, correlation_matrix,
#            orig_eq_nbr, eq_nbr, ramsey_eq_nbr, h, correlation_matrix_me,
#            endo, exo, param, orig_endo_nbr, lead_lag_incidence, nstatic, nfwrd,
#            npred, nboth, nsfwrd, nspred, ndynamic, equation_tags,
#            static_and_dynamic_models_differ, exo_names_orig_ord,
#            maximum_lag, maximum_lead, maximum_endo_lag,
#            maximum_endo_lead, maximum_exo_lead,
#            params,
            #            nnzderivatives,
#            static, dynamic,
#            user_written_analytical_steady_state, steady_state,
#            analytical_steady_state, static_params_derivs,
#            dynamic_params_derivs)
#            )
    end
end

end


