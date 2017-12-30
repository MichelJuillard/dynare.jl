module TestQ3a50
using Base.Test
using BenchmarkTools
using MAT

push!(LOAD_PATH,"../../src/models/")
push!(LOAD_PATH,"../models/q3a50/")

include("../../src/dynare.jl")
using .Dynare
using .Dynare.model

import .Dynare.DynLinAlg.Kronecker.a_mul_kron_b!

#cd("../models/q3a50")
#run(`/home/michel/dynare/git/master/dynare++/src/dynare++ --no-centralize q3a50.mod`)
#run(`/home/michel/dynare/git/master/matlab/preprocessor64/dynare_m q3a1.mod output=second language=julia`)
#cd("../../k_order")

vars = matread("../models/q3a50/q3a50.mat")

ss = vars["dyn_steady_states"]

g0 = vars["dyn_g_0"]
g1 = vars["dyn_g_1"]
g2 = vars["dyn_g_2"]

using q3a50

mod = q3a50.model_
endo_nbr = length(mod.endo)
exo_nbr = length(mod.exo)
lli = mod.lead_lag_incidence

m = Model(endo_nbr,lli,exo_nbr,0)

ipre = find(mod.lead_lag_incidence[1,:])
icur = find(mod.lead_lag_incidence[2,:])
ifwd = find(mod.lead_lag_incidence[3,:])
i1 = mod.lead_lag_incidence[3,ifwd]
i2 = mod.lead_lag_incidence[2,icur]

inverse_order_var, inverse_order_states = inverse_order_of_dynare_decision_rule(m)

ss = vars["dyn_steady_states"][inverse_order_var]
y = [ss[ipre]; ss[icur]; ss[ifwd]]
x = zeros(3,exo_nbr)
params = mod.params
steady_state = ss
it_ = 2
residual = zeros(endo_nbr)
nd = maximum(mod.lead_lag_incidence) + exo_nbr

steady_state = steady_state[:,1]
f = Array{Array{Float64,2}}(2)
f[1] = zeros(endo_nbr,nd)
f[2] = zeros(endo_nbr,nd^2)
mod.dynamic(y, x, params,steady_state, it_, residual,
            f[1], f[2])

order = 2

moments = Array{Array{Float64,1}}(2)
moments[1] = [0.0]
moments[2] = vec(eye(exo_nbr))

results_perturbation_ws = ResultsPerturbationWs(m,order)
algo = "CR"
first_order_ws =  FirstOrderSolverWS(algo, f[1], m)
type Cycle_Reduction
    tol
end
    
cr_opt = Cycle_Reduction(1e-8)

type Generalized_Schur
    criterium
end

gs_opt = Generalized_Schur(1+1e-6)

type Options
    cycle_reduction
    generalized_schur
end

options = Options(cr_opt,gs_opt)
first_order_solver(results_perturbation_ws, first_order_ws, algo, f[1], m, options)

g1_endo = copy(g1[:, 1:endo_nbr])
vg1_endo = view(g1_endo,inverse_order_var, inverse_order_states)
n_states = m.n_bkwrd + m.n_both
@test vg1_endo ≈ results_perturbation_ws.g[1][:,1:n_states]
@test g1[inverse_order_var, n_states + (1:exo_nbr)] ≈ results_perturbation_ws.g[1][:, n_states + (1:exo_nbr)]

k_order_ws = KOrderWs(endo_nbr,length(ifwd),length(ipre),endo_nbr,exo_nbr,ifwd,ipre,collect(1:endo_nbr),1:length(ipre),2)

k_order_solution!(results_perturbation_ws.g, f, moments, 2, k_order_ws)
println("timing k_order_solution!")
@time k_order_solution!(results_perturbation_ws.g, f, moments, 2, k_order_ws)

g2a = Array{Float64}(endo_nbr, n_states + exo_nbr + 1, n_states + exo_nbr + 1)

for i = 1:n_states + exo_nbr
    if i <= n_states
        col1 = inverse_order_states[i]
    else
        col1 = i
    end
    for j = i:n_states + exo_nbr
        if j <= n_states
            col2 = inverse_order_states[j]
        else
            col2 = j
        end
        if col1 > col2
            c1 = col2
            c2 = col1
        else
            c1 = col1
            c2 = col2
        end
        col = round(Int64,(n_states + exo_nbr + 1)*(n_states + exo_nbr)/2 - (n_states + exo_nbr - c1 + 2)*(n_states + exo_nbr - c1 + 1)/2 + c2 - c1 + 1)
        for k = 1:endo_nbr
            g2a[k, i, j] = 2*g2[inverse_order_var[k], col]
            g2a[k, j, i] = g2a[k, i, j]
        end
    end
end
g2b = reshape(g2a, endo_nbr, (n_states + exo_nbr + 1)^2)

g2b[:,end] = 2*g0[inverse_order_var]
        
@test g2b[1:3,1:3] ≈ results_perturbation_ws.g[2][1:3,1:3]

using Base.Profile
#@profile k_order_solution!(g,f,moments,2,k_order_ws)
#@time k_order_solution!(g,f,moments,2,k_order_ws)


end
