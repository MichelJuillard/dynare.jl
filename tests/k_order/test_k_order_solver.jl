include("../../src/taylor/faadibruno.jl")
include("../../src/linalg/linalg.jl")
include("../../src/solvers/solve_eye_plus_at_kron_b_optim.jl")
include("../../src/solvers/k_order_solver.jl")
include("../models/burnside.jl")
module Tst
using KOrder
using Base.Test
using ForwardDiff
using BurnsideModel

# variables order: x y

order = 4
nvar = 2
nfwrd = 2
nstate = 1
ncur = 2
nshock = 1
fwrd_index = [1, 2] 
state_index = [1]
cur_index = collect(1:2)
ng = nstate + nshock + 1

g1 = [rho 1 0; dgx*rho dgx 0]
g2 = [zeros(1,9); dgx2*rho^2 dgx2*rho 0 dgx2*rho dgx2 0 0 0 dgs2]
g3 = [zeros(3,9)
      dgx3*rho^3 dgx3*rho^2 0 dgx3*rho^2 dgx3*rho 0 0 0 dgx*rho*dgs2
      dgx3*rho^2 dgx3*rho   0 dgx3*rho   dgx3     0 0 0 dgx*dgs2
      0     0    dgx*rho*dgs2 0 0 dgx*dgs2          dgx*rho*dgs2 dgx*dgs2 0
      ]
g3 = reshape(g3,2,27)
g4 = [zeros(9,9)
      dgx4*rho^4 dgx4*rho^3 0 dgx4*rho^3 dgx4*rho^2 0 0 0 dgx2*rho^2*dgs2
      dgx4*rho^3 dgx4*rho^2 0 dgx4*rho^2 dgx4*rho   0 0 0 dgx2*rho*dgs2
      0    0   dgx2*rho^2*dgs2 0    0    dgx2*rho*dgs2 dgx2*rho^2*dgs2 dgx2*rho*dgs2 0
      dgx4*rho^3 dgx4*rho^2 0 dgx4*rho^2 dgx4*rho 0 0 0 dgx2*rho*dgs2
      dgx4*rho^2 dgx4*rho   0 dgx4*rho   dgx4     0 0 0 dgx2*dgs2
      0     0   dgx2*rho*dgs2 0 0 dgx2*dgs2         dgx2*rho*dgs2 dgx2*dgs2 0
      0 0 dgx2*rho^2*dgs2     0 0 dgx2*rho*dgs2       dgx2*rho^2*dgs2 dgx2*rho*dgs2 0
      0 0 dgs2*rho*dgs2       0 0 dgx2*dgs2           dgx2*rho*dgs2 dgx2*dgs2 0
      dgx2*rho^2*dgs2 dgx2*rho*dgs2 0 dgx2*rho*dgs2 dgx2*dgs2 0 0 0 dgs4
      ]
g4 = reshape(g4,2,81)
g = [g1, g2, g3, g4]
g_target = deepcopy(g)
mgg = ng
ngg = ng + nshock
gg = [zeros(mgg,ngg^i) for i = 1:order]

k_order_ws = KOrderWs(nvar, nfwrd, nstate, ncur, nshock, fwrd_index, state_index, cur_index, 1:nstate, order)

make_gg!(gg,g,1,k_order_ws)
target = vcat(hcat(g[1][state_index,:], zeros(nstate,nshock)),
              hcat(zeros(nshock, ng), eye(nshock)),
              hcat(zeros(1, nstate + nshock), 1, zeros(1,nshock)))
@test gg[1] == target

order = 2
make_gg!(gg,g,order,k_order_ws)
k = [ i+j*(ng+1) for j=0:(ng-1) for i=1:ng]
@test gg[2][1:nstate,k] ≈ g[2][state_index,:]
k = [ng+i + j*(ng+1) for j=0:(ng-1) for i=1:nshock]
@test gg[2][1:nstate,k] == zeros(nstate,ng*nshock)
@test gg[2][1:nstate,ng*(ng+nshock)+collect(1:(ng+nshock))] == zeros(nstate,ng+nshock)
make_gg!(gg,g,3,k_order_ws)

make_gg!(gg,g,4,k_order_ws)

mhh = nfwrd + nvar + nstate + nshock
nhh = nstate + 2*nshock + 1
hh = [zeros(mhh,nhh^i) for i = 1:4]
make_hh!(hh, g, gg, 1, k_order_ws)
@test size(hh[1]) == (nfwrd + nvar + nstate + nshock, nstate + 2*nshock + 1)
target0 = g[1][fwrd_index,1:nstate]*g[1][state_index,:]
target0[:, nstate + nshock + 1] += g[1][fwrd_index, nstate + nshock + 1] 
target = vcat(hcat(target0, g[1][fwrd_index,nstate+(1:nshock)]),
              hcat(g[1],zeros(nvar,nshock)),
              hcat(eye(nstate+nshock), zeros(nstate+nshock,nshock+1)))
@test hh[1] == target

make_hh!(hh, g, gg, 2, k_order_ws)
@test size(hh[1]) == (nfwrd + nvar + nstate + nshock, nstate + 2*nshock + 1) 
@test size(hh[2]) == (nfwrd + nvar + nstate + nshock, (nstate + 2*nshock + 1)^2)
k = [i+j*(nstate+nshock+1) for j in 0:(nstate-1) for i in 1:nstate ]
m = [i+j*(nstate+2*nshock+1) for j in 0:(nstate+nshock) for i in 1:(nstate+nshock+1)]
t1 = g[2][fwrd_index,k]*kron(g[1][state_index,:],g[1][state_index,:])
t2 = g[1][fwrd_index,1:nstate]*g[2][state_index,:]
v1 = zeros(nfwrd,(nstate + 2*nshock + 1)^2)
v1[:,m] = t1 + t2
v1[:,4] = g[2][fwrd_index,2]*g[1][1,1]
v1[:,8] = g[2][fwrd_index,2]*g[1][1,2]
v1[:,13] = v1[:,4]
v1[:,14] = v1[:,8]
v1[:,16] = g[2][fwrd_index,5]
v1[:,11] = g[1][fwrd_index,1]*g[2][1,9] + g[2][fwrd_index,9]
v2 = zeros(nvar,(nstate + 2*nshock + 1)^2)
v2[:,m] = g[2]
target = vcat(v1,
              v2,
              zeros(nstate+nshock, (nstate + 2*nshock + 1)^2))
@test hh[2] ≈ target

g[2] = zeros(Float64,2,9)
moments = [[0],[sigma2]]
k_order_ws_1 = KOrderWs(nvar, nfwrd, nstate, ncur, nshock, fwrd_index, state_index, cur_index, 1:nstate, order)
k_order_solution!(g,df,moments,order,k_order_ws_1)
@test g[2] ≈ g_target[2]
println("")

end
