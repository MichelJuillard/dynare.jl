module TestQ3a1
using Base.Test
using BenchmarkTools
using MAT

push!(LOAD_PATH,"../../src/models/")
push!(LOAD_PATH,"../models/q3a1/")

include("../../src/dynare.jl")
using .Dynare
using .Dynare.model

import .Dynare.DynLinAlg.Kronecker.a_mul_kron_b!


cd("../models/q3a1")
run(`/home/michel/dynare/git/master/dynare++/src/dynare++ --no-centralize q3a1.mod`)
run(`/home/michel/dynare/git/master/matlab/preprocessor64/dynare_m q3a1.mod output=second language=julia`)
cd("../../k_order")

vars = matread("../models/q3a1/q3a1.mat")

ss = vars["dyn_steady_states"]
g1 = vars["dyn_g_1"]
g2 = vars["dyn_g_2"]

using q3a1

mod = q3a1.model_
endo_nbr = length(mod.endo)
exo_nbr = length(mod.exo)
lli = mod.lead_lag_incidence

m = Model(endo_nbr,lli,exo_nbr,0)


ipre = find(mod.lead_lag_incidence[1,:])
icur = find(mod.lead_lag_incidence[2,:])
ifwd = find(mod.lead_lag_incidence[3,:])
i1 = mod.lead_lag_incidence[3,ifwd]
i2 = mod.lead_lag_incidence[2,icur]
ss = ss[:,1]
ss = ss[[3, 2, 1]]
y = [ss[ipre]; ss[icur]; ss[ifwd]]
x = zeros(3,exo_nbr)
params = mod.params
steady_state = ss
println(steady_state)
it_ = 2
residual = zeros(endo_nbr)
nd = maximum(mod.lead_lag_incidence) + exo_nbr

f = Array{Array{Float64,2}}(2)
f[1] = zeros(endo_nbr,nd)
f[2] = zeros(endo_nbr,nd^2)
mod.dynamic(y, x, params,steady_state, it_, residual,
            f[1], f[2])

order = 2

moments = Array{Array{Float64,1}}(2)
moments[1] = [0.0]
moments[2] = vec(mod.sigma_e)

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

inverse_order_var, inverse_order_states = inverse_order_of_dynare_decision_rule(m)
g1_endo = copy(g1[:, 1:endo_nbr])
vg1_endo = view(g1_endo,inverse_order_var, inverse_order_states)
n_states = m.n_bkwrd + m.n_both
@test vg1_endo ≈ results_perturbation_ws.g[1][:,1:n_states]
@test g1[inverse_order_var, n_states + (1:exo_nbr)] ≈ results_perturbation_ws.g[1][:, n_states + (1:exo_nbr)]

k_order_ws = KOrderWs(endo_nbr,length(ifwd),length(ipre),endo_nbr,exo_nbr,ifwd,ipre,collect(1:endo_nbr),1:length(ipre),2)

k_order_solution!(results_perturbation_ws.g, f, moments, 2, k_order_ws)

display(g2)
println(" ")
g2a = Array{Float64}(endo_nbr, n_states + exo_nbr + 1, n_states + exo_nbr + 1)

col = 1
for i = 1:n_states + exo_nbr
    println(" ")
    for j = i:n_states + exo_nbr
        println(col)
        for k = 1:endo_nbr
            g2a[k, i, j] = g2[k,col]
            g2a[k, j, i] = g2a[k, i, j]
        end
        col += 1
    end
end
g2b = reshape(g2a, endo_nbr, (n_states + exo_nbr + 1)^2)

@test g2b ≈ results_perturbation_ws.g[2]

using Base.Profile
#@profile k_order_solution!(g,f,moments,2,k_order_ws)
#@time k_order_solution!(g,f,moments,2,k_order_ws)


    
if false                     
    a = zeros(endo_nbr,endo_nbr)
    b = zeros(endo_nbr,endo_nbr)
    a = jacobian[:,i2]
    a[:,ipre] += jacobian[:,i1]*first_order_ws.ghx[ifwd,:]
    b[:,icur] = jacobian[:,i2]
    c = first_order_ws.ghx[ipre,:]
    #c = randn(length(ipre),length(ipre))
    nd1 = maximum(mod.lead_lag_incidence)
    K = reshape(1:nd1*nd1,nd1,nd1)
    gg = first_order_ws.ghx[ifwd,:]*first_order_ws.ghx[ipre,:]

    d = zeros(endo_nbr,nd1*nd1)
    order = 2
    aa = g2[:,vec(K)]
    bb = [gg;first_order_ws.ghx;eye(m.n_bkwrd+m.n_both)]
    a_mul_kron_b!(d, aa, bb, order)
    ws = EyePlusAtKronBWS(a,b,order,c)

    vd = view(d,:,1:(m.n_bkwrd+m.n_both)^2)
    general_sylvester_solver!(a,b,c,vd,order,ws)
end

end
