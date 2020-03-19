module DynareModel

export dynare_model, EquationTag

mutable struct Endo
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

mutable struct Exo
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

mutable struct Param
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

mutable struct Temporaries
    static::Vector{Int64}
    dynamic::Vector{Int64}
    function Temporaries()
        static = Vector{Int64}(undef, 4)
        dynamic = Vector{Int64}(undef, 4)
        new(static, dynamic)
    end
end

mutable struct EquationTag
    equation::Int64
    name::String
    value::String
end

mutable struct Mapping
    eqidx::Dict{String, Vector{Int64}}
end

mutable struct dynare_model
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
    endo_nbr::Int64
    exo_nbr::Int64
    param_nbr::Int64
    nstatic::Int64
    nfwrd::Int64
    npred::Int64
    nboth::Int64
    nsfwrd::Int64
    nspred::Int64
    ndynamic::Int64
    equation_tags::Vector{EquationTag}
    static_and_dynamic_models_differ::Bool
    has_external_function::Bool
    exo_names_orig_ord::Vector{Int64}
    maximum_lag::Int64
    maximum_lead::Int64
    maximum_endo_lag::Int64
    maximum_endo_lead::Int64
    maximum_exo_lag::Int64
    maximum_exo_lead::Int64
    orig_maximum_endo_lag::Int64
    orig_maximum_endo_lead::Int64
    orig_maximum_exo_lag::Int64
    orig_maximum_exo_lead::Int64
    orig_maximum_exo_det_lag::Int64
    orig_maximum_exo_det_lead::Int64
    orig_maximum_lag::Int64
    orig_maximum_lead::Int64
    orig_maximum_lag_with_diffs_expanded::Int64
    params::Vector{Float64}
    nnzderivatives::Vector{Int64}
    static::Function
    dynamic::Function
    temporaries::Temporaries
    user_written_analytical_steady_state::Bool
    steady_state::Function
    analytical_steady_state::Bool
    static_params_derivs::Function
    dynamic_params_derivs::Function
    state_var::Vector{Int64}
    mapping::Mapping
    
    function dynare_model()
        fname = ""
        dynare_version = ""
        sigma_e = Matrix{Float64}(undef, 0, 0)
        correlation_matrix = Matrix{Float64}(undef, 0, 0)
        orig_eq_nbr = 0
        eq_nbr = 0
        ramsey_eq_nbr = 0
        h = Matrix{Float64}(undef, 0,0)
        correlation_matrix_me = Matrix{Float64}(undef, 0, 0)
        endo = []
        endo_nbr = 0
        exo = []
        exo_nbr = 0
        param = []
        param_nbr = 0
        orig_endo_nbr = 0
        lead_lag_incidence = Matrix{Int64}(undef, 0, 0)
        orig_maximum_endo_lag = 0
        nstatic = 0
        nfwrd = 0
        npred = 0
        nboth = 0
        nsfwrd = 0
        nspred = 0
        ndynamic = 0
        equation_tags = []
        static_and_dynamic_models_differ = false
        has_external_function = false
        exo_names_orig_ord = []
        maximum_lag = 0
        maximum_lead = 0
        maximum_endo_lag = 0
        maximum_endo_lead = 0
        maximum_exo_lag = 0
        maximum_exo_lead = 0
        orig_maximum_endo_lag = 0
        orig_maximum_endo_lead = 0
        orig_maximum_exo_lag = 0
        orig_maximum_exo_lead = 0
        orig_maximum_exo_det_lag = 0
        orig_maximum_exo_det_lead = 0
        orig_maximum_lag = 0
        orig_maximum_lead = 0
        orig_maximum_lag_with_diffs_expanded = 0
        params = []
        nnzderivatives = []
        f_nothing() = nothing
        static = f_nothing
        dynamic = f_nothing
        temporaries = DynareModel.Temporaries()
        user_written_analytical_steady_state = false
        steady_state = f_nothing
        analytical_steady_state = false
        static_params_derivs = f_nothing
        dynamic_params_derivs = f_nothing
        state_var = []
        mapping = Mapping(Dict())
        new(fname, dynare_version, sigma_e, correlation_matrix,
            orig_eq_nbr, eq_nbr, ramsey_eq_nbr, h, correlation_matrix_me,
            endo, exo, param, orig_endo_nbr, lead_lag_incidence, endo_nbr,
            exo_nbr, param_nbr, nstatic, nfwrd, npred, nboth, nsfwrd,
            nspred, ndynamic, equation_tags, static_and_dynamic_models_differ,
            has_external_function,
            exo_names_orig_ord, maximum_lag, maximum_lead, maximum_endo_lag,
            maximum_endo_lead, maximum_exo_lag, maximum_exo_lead,
            orig_maximum_endo_lag, orig_maximum_endo_lead, orig_maximum_exo_lag,
            orig_maximum_exo_lead, orig_maximum_exo_det_lag,
            orig_maximum_exo_det_lead, orig_maximum_lag, orig_maximum_lead,
            orig_maximum_lag_with_diffs_expanded, params, nnzderivatives,
            static, dynamic, temporaries, user_written_analytical_steady_state,
            steady_state, analytical_steady_state, static_params_derivs,
            dynamic_params_derivs, state_var, mapping)
    end
end

end


