push!(LOAD_PATH,".")
push!(LOAD_PATH,"..")
push!(LOAD_PATH,"../linalg")
push!(LOAD_PATH,"../taylor")

using Base.Test
using Dynare
import Tmp

nendo = 3
nfwrd = 2
nstate = 2
nshock = 2
fwrd_index = [1, 3]
state_index = [1, 2]

k = 1
i = 1
order = k + i
g = randn(nendo, (nstate + nshock + 1)^order)
gsσi = randn(nstate, (nstate + nshock)^k)

Tmp.make_gsσi!(gsσi, g, nstate, nshock, state_index, k, i)
@test gsσi == g[state_index, 5:5:20]

println("Test")
k = 2
i = 2
order = k + i
g = randn(nendo, (nstate + nshock + 1)^order)
gsσi = randn(nstate, (nstate + nshock)^k)

cols = reshape(collect(25:25:25^2),5,5)
cols = cols[1:4, 1:4]
Tmp.make_gsσi!(gsσi, g, nstate, nshock, state_index, k, i)
@test gsσi == g[state_index, vec(cols)]

maxorder = 5
nn = nstate + nshock + 1
g = [randn(nendo, nn^i) for i = 1:maxorder]
Sigma = [randn(nshock^l) for l = 1:maxorder]
gfykσlΣ = [zeros(nfwrd, Dynare.Solvers.KOrderSolver.number_of_unique_derivatives(k,nstate)) for k = 1:maxorder]

println("D02")
k = 0
j = 2
order = k + j
gul = rand(nfwrd, nshock^order)
dkj = zeros(nfwrd, (nstate + nshock)^k)
dkji = similar(dkj)
gsσ = [randn(nstate, (nstate + nshock)^i) for i in 0:k]
faa_di_bruno_ws = Dynare.FaaDiBruno.FaaDiBrunoWs(nfwrd, nstate, order)
Tmp.make_dkj!(dkj, g, k, j, gfykσlΣ, Sigma, nstate, nfwrd, nshock, fwrd_index, state_index, gul, faa_di_bruno_ws, dkji, gsσ)
@test dkj ≈ g[2][fwrd_index, [13, 14, 18, 19]]*Sigma[2]

println("D12")
k = 1
j = 2
order = k + j
gul = rand(nfwrd, nshock^order)
dkj = zeros(nfwrd, (nstate + nshock)^k)
dkji = similar(dkj)
gsσ = [randn(nstate, (nstate + nshock)^i) for i in 1:k]
faa_di_bruno_ws = Dynare.FaaDiBruno.FaaDiBrunoWs(nfwrd, nstate, order)
Tmp.make_dkj!(dkj, g, 1, 2, gfykσlΣ, Sigma, nstate, nfwrd, nshock, fwrd_index, state_index, gul, faa_di_bruno_ws, dkji, gsσ)
X = zeros(nfwrd, nstate)
cols = [13, 14, 18, 19]
for i = 1:nstate
    X[:,i] = g[3][fwrd_index, cols]*Sigma[2]
    cols += (nstate + nshock + 1)^2
end
Y = g[1][state_index, 1:4]
@test dkj ≈ X*Y
