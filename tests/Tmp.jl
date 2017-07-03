module Tmp
using BenchmarkTools
using MAT

include("q3a50.jl")
include("../src/dynare.jl")
using .Dynare

import .Dynare.LinAlg.Kronecker.a_mul_kron_b!

vars = matread("q3a50.mat")

ss = vars["dyn_steady_states"]


#import .FirstOrder: FirstOrderSolverWS, first_order_solver

mod = Tmp.q3a50.model_
endo_nbr = length(mod.endo)
exo_nbr = length(mod.exo)
lli = mod.lead_lag_incidence

m = Model(endo_nbr,lli)

vars = matread("q3a50.mat")
ss = vec(vars["dyn_steady_states"])

ipre = find(mod.lead_lag_incidence[1,:])
icur = find(mod.lead_lag_incidence[2,:])
ifwd = find(mod.lead_lag_incidence[3,:])
i1 = mod.lead_lag_incidence[3,ifwd]
i2 = mod.lead_lag_incidence[2,icur]
y = [ss[ipre]; ss[icur]; ss[ifwd]]
x = zeros(3,exo_nbr)
params = mod.params
steady_state = ss
it_ = 2
residual = zeros(endo_nbr)
nd = maximum(mod.lead_lag_incidence) + exo_nbr
g1 = zeros(mod.eq_nbr,nd)
g2 = zeros(mod.eq_nbr,nd*nd)

mod.dynamic(y, x, params,steady_state, it_, residual,
            g1, g2)

jacobian = copy(g1)

d,e = get_de(g1,m)

algo = "CR"
first_order_ws =  FirstOrderSolverWS(algo, g1, m)
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
first_order_solver(first_order_ws,algo, g1, m, options)


second_order_ws = SecondOrderSolverWS(endo_nbr,exo_nbr, m)

second_order_solver!(jacobian, g2, m, first_order_ws, second_order_ws)

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
