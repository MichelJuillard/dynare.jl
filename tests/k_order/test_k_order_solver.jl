include("../../src/solvers/k_order_solver.jl")
using KOrder
using Base.Test

nvar = 3
nstate = 2
nshock = 1
state_index = [1, 3]
ng = nstate + nshock + 1
g = [randn(nvar,ng^i) for i = 1:4]
mgg = ng + nstate + nshock 
ngg = ng + nshock
gg = [zeros(mgg,ngg^i) for i = 1:4]

k_order_ws = KOrderWs(nstate, nshock, state_index, 1:nstate)

make_gg!(gg,g,1,k_order_ws)
@test gg[1] == vcat(hcat(g[1][state_index,:], zeros(nstate,nshock)), eye(ng + nshock))

make_gg!(gg,g,2,k_order_ws)

@test gg[2][1:nstate,1:4] == g[2][state_index,1:4]
@test gg[2][1:nstate,6:9] == g[2][state_index,5:8]
@test gg[2][1:nstate,11:14] == g[2][state_index,9:12]
@test gg[2][1:nstate,16:19] == g[2][state_index,13:16]

make_gg!(gg,g,3,k_order_ws)
@test gg[3][1:nstate,1:4] == g[3][state_index,1:4]
@test gg[3][1:nstate,6:9] == g[3][state_index,5:8]
@test gg[3][1:nstate,11:14] == g[3][state_index,9:12]
@test gg[3][1:nstate,16:19] == g[3][state_index,13:16]
@test gg[3][1:nstate,26:29] == g[3][state_index,17:20]
@test gg[3][1:nstate,31:34] == g[3][state_index,21:24]
@test gg[3][1:nstate,36:39] == g[3][state_index,25:28]
@test gg[3][1:nstate,41:44] == g[3][state_index,29:32]
@test gg[3][1:nstate,51:54] == g[3][state_index,33:36]
@test gg[3][1:nstate,56:59] == g[3][state_index,37:40]
@test gg[3][1:nstate,61:64] == g[3][state_index,41:44]
@test gg[3][1:nstate,66:69] == g[3][state_index,45:48]
@test gg[3][1:nstate,76:79] == g[3][state_index,49:52]
@test gg[3][1:nstate,81:84] == g[3][state_index,53:56]
@test gg[3][1:nstate,86:89] == g[3][state_index,57:60]
@test gg[3][1:nstate,91:94] == g[3][state_index,61:64]
@test gg[3][:,95:125] == zeros(mgg,31)

make_gg!(gg,g,4,k_order_ws)
@test gg[4][1:nstate,1:4] == g[4][state_index,1:4]
@test gg[4][1:nstate,26:29] == g[4][state_index,17:20]
@test gg[4][1:nstate,51:54] == g[4][state_index,33:36]
@test gg[4][1:nstate,76:79] == g[4][state_index,49:52]
@test gg[4][1:nstate,126:129] == g[4][state_index,65:68]
@test gg[4][1:nstate,251:254] == g[4][state_index,129:132]
@test gg[4][1:nstate,376:379] == g[4][state_index,193:196]
@test gg[4][1:nstate,466:469] == g[4][state_index,253:256]
@test gg[4][1:nstate,500:625] == zeros(nstate,126)
