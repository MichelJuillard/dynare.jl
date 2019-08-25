push!(LOAD_PATH,"../../src/")
push!(LOAD_PATH,"../models/")

module TestKOrderSolver
using Dynare
import Dynare.Solvers.KOrderSolver: KOrderWs, make_gg!, make_hh!, k_order_solution!, make_rhs_1!, make_rhs_2!, store_results_1!, make_gs_su!, make_gykf!, compute_derivatives_wr_shocks!, make_a1!, generalized_sylvester_solver!, store_results_2!, make_gsk!
import Dynare.FaaDiBruno: FaaDiBrunoWs, partial_faa_di_bruno!
import Dynare.DynLinAlg.LinSolveAlgo: LinSolveWS
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

function test_burnside(f, g, moments, ws, order)
    gg = ws.gg::Vector{Matrix{Float64}}
    hh = ws.hh::Vector{Matrix{Float64}}
    rhs = ws.rhs::Matrix{Float64}
    rhs1 = ws.rhs1::Matrix{Float64}
    faa_di_bruno_ws = ws.faa_di_bruno_ws_2::FaaDiBrunoWs
    nfwrd = ws.nfwrd::Int64
    fwrd_index = ws.fwrd_index::Vector{Int64}
    nstate = ws.nstate::Int64
    state_index = ws.state_index::Vector{Int64}
    ncur = ws.ncur::Int64
    cur_index = ws.cur_index::Vector{Int64}
    nvar = ws.nvar::Int64
    nshock = ws.nshock::Int64
    a = ws.a::Matrix{Float64}
    b = ws.b::Matrix{Float64}
    linsolve_ws = ws.linsolve_ws_1::LinSolveWS
    work1 = ws.work1::Vector{Float64}
    work2 = ws.work2::Vector{Float64}
    gs_ws = ws.gs_ws::EyePlusAtKronBWS
    gs_ws_result = gs_ws.result::Matrix{Float64}

    make_gg!(gg, g, order - 1, ws)

    if order > 1
        target = vcat(hcat(g[1][state_index,:], zeros(nstate,nshock)),
                      hcat(zeros(nshock, ng), eye(nshock)),
                      hcat(zeros(1, nstate + nshock), 1, zeros(1,nshock)))
        @test gg[1] == target
    elseif order > 2
        k = [ i+j*(ng+1) for j=0:(ng-1) for i=1:ng]
        @test gg[2][1:nstate,k] ≈ g[2][state_index,:]
    elseif order > 3
        k = [ng+i + j*(ng+1) for j=0:(ng-1) for i=1:nshock]
        @test gg[2][1:nstate,k] == zeros(nstate,ng*nshock)
        @test gg[2][1:nstate,ng*(ng+nshock)+collect(1:(ng+nshock))] == zeros(nstate,ng+nshock)
    end

    make_hh!(hh, g, gg, order - 1, ws)

    if order > 1
        @test size(hh[1]) == (nstate + nvar + nfwrd + nshock, nstate + 2*nshock + 1)
        target0 = g[1][fwrd_index,1:nstate]*g[1][state_index,:]
        target0[:, nstate + nshock + 1] += g[1][fwrd_index, nstate + nshock + 1] 
        target = vcat(hcat(eye(nstate), zeros(nstate,2*nshock+1)),
                      hcat(g[1],zeros(nvar,nshock)),
                      hcat(target0, g[1][fwrd_index,nstate+(1:nshock)]),
                      hcat(zeros(nshock,nstate), eye(nshock), zeros(nshock, nshock + 1)))
        @test hh[1] == target
    elseif order > 2
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
        target = vcat(zeros(nstate, (nstate + 2*nshock + 1)^2),
                      v2,
                      v1,
                      zeros(nshock, (nstate + 2*nshock + 1)^2))
        @test hh[2] ≈ target
    end

    partial_faa_di_bruno!(rhs,f,hh,order,faa_di_bruno_ws)
    @test rhs ≈ f[2]*kron(hh[1],hh[1])
    
    if order > 1
        make_a1!(a, f, g, ncur, cur_index, nvar, nstate, nfwrd, fwrd_index, state_index)
        AA = f[1][:,2:3]
        AA[:,1] += f[1][:,4:5]*g1[:,1]
        @test a ≈ AA
    end

    @inbounds for i = 1:nfwrd
        col1 = fwrd_index[i]
        col2 = nstate + ncur + i
        @simd for j=1:nvar
            b[j, col1] = f[1][j, col2]
        end
    end
    c = g[1][state_index,1:nstate]

    make_rhs_1!(rhs1, rhs, nstate, nshock, nvar, order)
    @test rhs1[:, 1] == -rhs[:, 1]
    
    d = Vector{Float64}(nvar*nstate^order)
    copy!(d, 1, rhs1, 1, nvar*nstate^order)
    @test b == f[1][:,4:5]
    @test c == g1[1:1,1:1]
    generalized_sylvester_solver!(a,b,c,d,order,gs_ws)
    store_results_1!(g[order], gs_ws_result, nstate, nshock, nvar, order)
    @test g[2][:, 1] ≈ g2[:, 1]
    
    compute_derivatives_wr_shocks!(ws,f,g,order)
    if order == 2
        @test ws.rhs1[:, 1:2] ≈ g2[:, [2, 5]]
    elseif order == 3
        @test ws.rhs1[:, 1:2] ≈ g3[:, [2, 5]]
    end

    store_results_2!(g[order], nstate, nshock, nvar, rhs1, order)
    @test g[2][:, [2, 4, 5]] ≈ g2[:, [2, 4, 5]]
    make_gsk!(g, f, moments[2], a, rhs, rhs1,
              nfwrd, nstate, nvar, ncur, nshock,
              fwrd_index, linsolve_ws, work1, work2)
    if order == 2
        @test g[2][:,end] ≈ g2[:,end]
    elseif order == 3
        @test g[3][:, 25:27] ≈ g2[:, 25:27]
    end
end

order = 2
g[1] = g1
g[2] = zeros(Float64,2,9)
moments = [[0],[sigma2]]

ws = KOrderWs(nvar, nfwrd, nstate, ncur, nshock, fwrd_index, state_index, cur_index, 1:nstate, order)
test_burnside(df, g, moments, ws, order)

if false
order = 2
g[2] = zeros(Float64,2,9)
moments = [[0],[sigma2]]
k_order_ws_1 = KOrderWs(nvar, nfwrd, nstate, ncur, nshock, fwrd_index, state_index, cur_index, 1:nstate, order)
k_order_solution!(g,df,moments,order,k_order_ws_1)
#@test g[2] ≈ g_target[2]
println("")

end

nvar1 = 3
nstate1 = 2
nshock1 = 2
order = 3
rhs = rand(nvar1, (nstate1 + 2*nshock1 + 1)^order)
rhs1 = rand(nvar1, nstate1^order)
make_rhs_1!(rhs1, rhs, nstate1, nshock1, nvar1, order)
k = [1, 2]
n = nstate1 + 2*nshock1 + 1 
k = vcat(k, k + n)
k = vcat(k, k + n^2)
@test rhs1 == -rhs[:,k]

order = 2
rhs = rand(nvar1, (nstate1 + 2*nshock1 + 1)^order)
rhs2 = rand(nvar1, nshock1*(nstate1 + nshock1)^(order-1))
rhs2_orig = copy(rhs2)
make_rhs_2!(rhs2, rhs, nstate1, nshock1, nvar1, order)
inc = nstate1 + 2*nshock1 + 1
k1 = [3, 4]
k = k1
for i = 1:nstate1 + nshock1 - 1
    k1 += inc
    k = vcat(k, k1)
end
@test rhs2 == -rhs2_orig - rhs[:,k]

order = 3
rhs = rand(nvar1, (nstate1 + 2*nshock1 + 1)^order)
rhs2 = rand(nvar1, nshock1*(nstate1 + nshock1)^(order-1))
rhs2_orig = copy(rhs2)
make_rhs_2!(rhs2, rhs, nstate1, nshock1, nvar1, order)
inc = nstate1 + 2*nshock1 + 1
k1 = [3, 4]
k2 = k1
for i = 1:nstate1 + nshock1 - 1
    k1 += inc
    k2 = vcat(k2, k1)
end
k = k2
for i = 1:nstate1 + nshock1 - 1
    k2 += inc^2
    k = vcat(k, k2)
end
@test rhs2 == -rhs2_orig - rhs[:,k]

x = rand(nvar1, nstate1^order)
results = rand(nvar1, (nstate1 + nshock1 + 1)^order)
store_results_1!(results, x, nstate1, nshock1, nvar1, order)
k = [1, 2]
n = nstate1 + nshock1 + 1 
k = vcat(k, k + n)
k = vcat(k, k + n^2)
@test results[:,k] == x


nshock1 = 10
nvar1 = 20
state_index1 = collect(1:2:nvar1)
nstate1 = length(state_index1)
x = randn(nvar1, nstate1 + nshock1 + 1)
gs_su = randn(nstate1, nstate1 + nshock1)
make_gs_su!(gs_su, x, nstate1, nshock1, state_index1)
@test gs_su == x[state_index1,1:(nstate1+nshock1)]

order = 3
nstate1 = 2
nshock1 = 3
nvar1 = 5
fwrd_index1 = collect(1:2:nvar1)
nfwrd1 = length(fwrd_index1)
x = randn(nvar1, (nstate1 + nshock1 + 1)^order)
gykf = randn(nfwrd1, nstate1^order)
make_gykf!(gykf, x, nstate1, nfwrd1, nshock1, fwrd_index1, order)
inc = nstate1 + nshock1 + 1
k1 = [1, 2]
k2 = k1
for i = 1:nstate1 - 1
    k1 += inc
    k2 = vcat(k2, k1)
end
k = k2
for i = 1:nstate1 - 1
    k2 += inc^2
    k = vcat(k, k2)
end
@test gykf == x[fwrd_index1, k]


end
