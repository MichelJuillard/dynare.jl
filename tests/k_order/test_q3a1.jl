include("../../src/set_path.jl")
module TestQ3a1
using Test
using BenchmarkTools
using LinearAlgebra
using MAT

#push!(LOAD_PATH,"../src/models/")
push!(LOAD_PATH,"models/q3a1/")

using Dynare
using FaaDiBruno
using KOrderSolver
using KroneckerUtils
using LinSolveAlgo
using SolveEyePlusMinusAkronB

function folded_linear_indices(c::CartesianIndices)
    order = ndims(c)
    s = size(c)
    if !all(s[1] .== s)
        error("all dimension must have same length")
    end
    fli = Array{Int64}(undef, size(c))
    k = 1
    for ci in c
        cci = collect(ci.I)
        scci = sort(cci, rev=true)
        if cci == scci
            fli[cci...] = k
            k += 1
        else
            fli[cci...] = fli[scci...]
        end
    end
    fli
end

CI = CartesianIndices((1:3,1:3))
@test folded_linear_indices(CI) == [1 2 3; 2 4 5; 3 5 6]

function unfold_dynarepp!(ug::Array{Float64,2}, g::Array{Float64,2}, nvar, order)
    CI = CartesianIndices(tuple(repeat([1:nvar],order)...))
    LI = LinearIndices(CI)
    ULI = folded_linear_indices(CI)
    k = 1
    copyto!(ug,g[:,vec(ULI)])
end

g = Array{Float64,2}([collect(1:10)'; collect(1:10)'])
ug = zeros(2,27)

unfold_dynarepp!(ug, g, 3, 3)

target_I = [1, 2, 3, 2, 4, 5, 3, 5, 6,
            2, 4, 5, 4, 7, 8, 5, 8, 9,
            3, 5, 6, 5, 8, 9, 6, 9, 10]
@test ug == g[:, target_I]

cd("models/q3a1")
run(`/home/michel/projects/dynare/git/master2/dynare++/src/dynare++ --no-centralize q3a1_1.mod`)
run(`/home/michel/projects/dynare/git/master2/dynare++/src/dynare++ --no-centralize q3a1_3.mod`)
run(`/home/michel/projects/dynare/git/master2/matlab/preprocessor64/dynare_m q3a1_3.mod output=third language=julia`)
cd("../..")

vars = matread("models/q3a1/q3a1_1.mat")
ss = vars["dyn_steady_states"]
g0_1 = vars["dyn_g_0"]
g1_1 = vars["dyn_g_1"]
#g2 = vars["dyn_g_2"]

vars = matread("models/q3a1/q3a1_3.mat")
g0_3 = vars["dyn_g_0"]
g1_3 = vars["dyn_g_1"]
g2_3 = vars["dyn_g_2"]
g3_3 = vars["dyn_g_3"]

using q3a1_3

mod = q3a1_3.model_
endo_nbr = length(mod.endo)
exo_nbr = length(mod.exo)
lli = mod.lead_lag_incidence

m = Model(endo_nbr,lli,exo_nbr,0)

ipre = findall(mod.lead_lag_incidence[1,:] .> 0)
icur = findall(mod.lead_lag_incidence[2,:] .> 0)
ifwd = findall(mod.lead_lag_incidence[3,:] .> 0)
i1 = mod.lead_lag_incidence[3,ifwd]
i2 = mod.lead_lag_incidence[2,icur]
ss = ss[:,1]
ss = ss[[3, 2, 1]]
y = [ss[ipre]; ss[icur]; ss[ifwd]]
x = zeros(3,exo_nbr)
params = mod.params
steady_state = ss

n1 = length(ipre) + exo_nbr
LI2 = LinearIndices((n1,n1))
ug2_3 = zeros(endo_nbr, length(LI2))
unfold_dynarepp!(ug2_3, g2_3, n1, 2)
inverse_order_var, inverse_order_states = inverse_order_of_dynare_decision_rule(m)
reorder = vcat(inverse_order_states, length(ipre) .+ (1:exo_nbr))
rLI2 = vec(LI2[ntuple(x->reorder,2)...])
r_ug2_3 = ug2_3[inverse_order_var, rLI2]

it_ = 2
residual = zeros(endo_nbr)
nd = maximum(mod.lead_lag_incidence) + exo_nbr

order = 3

f = Array{Array{Float64,2}}(undef, order)
f[1] = zeros(endo_nbr,nd)
f[2] = zeros(endo_nbr,nd^2)
f[3] = zeros(endo_nbr,nd^3)
temporary_terms = Vector{Float64}(undef, sum(mod.temporaries.dynamic[1:3]))
mod.dynamic(temporary_terms, residual, f[1], f[2], f[3], y, x, params, steady_state,
            it_)

moments = Array{Array{Float64,1}}(undef, order)
moments[1] = zeros(exo_nbr)
moments[2] = vec(Matrix{Float64}(I, exo_nbr, exo_nbr))
moments[3] = zeros(exo_nbr^3)

results_perturbation_ws = ResultsPerturbationWs(m,order)
algo = "CR"
first_order_ws =  FirstOrderSolverWS(algo, f[1], m)
struct Cycle_Reduction
    tol
end
    
cr_opt = Cycle_Reduction(1e-8)

struct Generalized_Schur
    criterium
end

gs_opt = Generalized_Schur(1+1e-6)

struct Options
    cycle_reduction
    generalized_schur
end

options = Options(cr_opt,gs_opt)
first_order_solver(results_perturbation_ws, first_order_ws, algo, f[1], m, options)

n_states = m.n_bkwrd + m.n_both
g1_endo = copy(g1_1[:, 1:(n_states + exo_nbr)])
vg1_endo = view(g1_endo,inverse_order_var, inverse_order_states)
@test vg1_endo ≈ results_perturbation_ws.g[1][:,1:n_states]
@test g1_1[inverse_order_var, n_states .+ (1:exo_nbr)] ≈ results_perturbation_ws.g[1][:, n_states .+ (1:exo_nbr)]

g = results_perturbation_ws.g
ws = KOrderWs(endo_nbr,length(ifwd),length(ipre),endo_nbr,exo_nbr,ifwd,ipre,collect(1:endo_nbr),1:length(ipre),order)

LIg = [LinearIndices(tuple(repeat([size(g[i],2)])...)) for i in 1:order]
LIgg = [LinearIndices(tuple(repeat([size(ws.gg[i],2)])...)) for i in 1:order]

KOrderSolver.make_gg!(ws.gg, g, 1, ws)
@test ws.gg[1] ≈ vcat(hcat(g[1][ws.state_index, :], zeros(ws.nstate, ws.nshock)),
                      hcat(zeros(ws.nshock+1, ws.nstate + ws.nshock), circshift(Matrix{Float64}(I, ws.nshock+1, ws.nshock+1), (0,1))))
nsrc = ws.nstate + ws.nshock + 1
ndest = ws.nstate + 2*ws.nshock + 1

order = 2

@testset "Order = 2" begin
    @testset "rhs 2"  begin
        KOrderSolver.make_gg!(ws.gg, g, order, ws)
        #@test ws.gg[2][1:ws.nstate, vec(D[2][1:nsrc,1:nsrc])] ≈ g[2][ws.state_index,:]
        #@test ws.gg[2][ws.nstate .+ (1:ws.nshock), 1:ndest] ≈ zeros(ws.nshock, ndest) 

        KOrderSolver.make_hh!(ws.hh, g, ws.gg, 1, ws)
        @test ws.hh[1][1:ws.nstate,1:ws.nstate] ≈ Matrix{Float64}(I, ws.nstate, ws.nstate)
        @test ws.hh[1][ws.nstate .+ (1:ws.ncur), 1:nsrc] ≈ g[1]
        @test ws.hh[1][ws.nstate + ws.ncur .+ ws.fwrd_index, 1:nsrc] ≈ g[1][ws.fwrd_index,1:ws.nstate]*g[1][ws.state_index,1:nsrc] 
        @test ws.hh[1][:,ws.nstate + ws.nshock + 1] ≈ zeros(ws.nfwrd + ws.ncur + ws.nstate + ws.nshock)            

        rhs = reshape(view(ws.rhs,1:ws.nvar*ndest^2),ws.nvar,ndest^2)
        FaaDiBruno.partial_faa_di_bruno!(rhs,f,ws.hh,order,ws.faa_di_bruno_ws_2)
        @test  rhs ≈ f[2]*kron(ws.hh[1], ws.hh[1])
    end
    
    @testset "states 2" begin
        rhs = reshape(view(ws.rhs,1:ws.nvar*ndest^2),ws.nvar,ndest^2)
        rhs1 = reshape(view(ws.rhs1,1:ws.nvar*ws.nstate^order),
                        ws.nvar, ws.nstate^order)
        KOrderSolver.pane_copy!(rhs1, rhs, 1:ws.nvar, 1:ws.nvar, 1:ws.nstate, 1:ws.nstate, ws.nstate, ws.nstate + 2*ws.nshock + 1, order)

        KOrderSolver.make_a!(ws.a, f, g, ws.ncur, ws.cur_index, ws.nvar, ws.nstate, ws.nfwrd, ws.fwrd_index, ws.state_index)
        target = zeros(ws.nvar,ws.nvar)
        target[:,ws.cur_index] = f[1][:,ws.nstate .+ ws.cur_index]
        target[:,ws.state_index] .+= f[1][:,ws.nstate + ws.ncur .+ (1:ws.nfwrd)]*g[1][ws.fwrd_index, 1:ws.nstate]
        @test ws.a ≈ target

        a_orig = copy(ws.a)
        KOrderSolver.make_b!(ws.b, f, ws)
        @test ws.b[:,ws.fwrd_index] ≈ f[1][:,ws.nstate + ws.ncur .+ (1:ws.nfwrd)]

        b_orig = copy(ws.b)
        c = view(g[1],ws.state_index,1:ws.nstate)
        c_orig = copy(c)
        ws.gs_ws = EyePlusAtKronBWS(ws.nvar,ws.nvar,ws.nstate,order)
        rhs1_orig = copy(rhs1)
        d = vec(rhs1)
        a = copy(ws.a)
        b = copy(ws.b)
        SolveEyePlusMinusAkronB.generalized_sylvester_solver!(a,b,c,d,order,ws.gs_ws)
        x = ws.gs_ws.result
        @test a_orig*x + b_orig*x*kron(c_orig,c_orig) ≈ rhs1_orig

        k_order_solution!(results_perturbation_ws.g, f, moments, order, ws)
        @test results_perturbation_ws.g[2][:, [1, 2, 6, 7]] ≈ 2*r_ug2_3[:,[1, 2, 5, 6]]
    end

    @testset "shocks 2" begin
        fp = view(f[1],:,ws.nstate + ws.ncur .+ (1:ws.nfwrd))

        KOrderSolver.make_gs_su!(ws.gs_su, g[1], ws.nstate, ws.nshock, ws.state_index)
        @test ws.gs_su == g[1][ws.state_index, 1:ws.nstate + ws.nshock]
        
        gykf = reshape(view(ws.gykf,1:ws.nfwrd*ws.nstate^order),
                       ws.nfwrd,ws.nstate^order)
        KOrderSolver.make_gykf!(gykf, g[order], ws.nstate, ws.nfwrd, ws.nshock, ws.fwrd_index, order)
        @test gykf ≈ g[2][ws.fwrd_index, vcat(1:ws.nstate, ws.nstate + ws.nshock +1 .+ (1:ws.nstate))]
            
        gu = view(ws.gs_su,:,ws.nstate .+ (1:ws.nshock))
        rhs1 = reshape(view(ws.rhs1,:1:ws.nvar*(ws.nshock*(ws.nstate+ws.nshock))^(order-1)),
                       ws.nvar,(ws.nshock*(ws.nstate+ws.nshock))^(order-1))
        work1 = view(ws.work1,1:ws.nvar*(ws.nstate + ws.nshock + 1)^order)
        work2 = view(ws.work2,1:ws.nvar*(ws.nstate + ws.nshock + 1)^order)
        a_mul_b_kron_c_d!(rhs1,fp,gykf,gu,ws.gs_su,order,work1,work2)
        @test rhs1 ≈ fp*gykf*kron(gu,ws.gs_su)
            
        rhs = reshape(view(ws.rhs,1:ws.nvar*(ws.nstate+2*ws.nshock+1)^order),
                      ws.nvar,(ws.nstate+2*ws.nshock+1)^order)
        rhs1_orig = copy(rhs1)
        KOrderSolver.make_rhs_2!(rhs1, rhs, ws.nstate, ws.nshock, ws.nvar)
        @test rhs1 == rhs[:,vcat(15:18, 22:25)] - rhs1_orig

        rhs1_orig = copy(rhs1)
        linsolve_core!(ws.linsolve_ws_1,Ref{UInt8}('N'),ws.a,rhs1)
        @test rhs1 ≈ ws.a\rhs1_orig

        KOrderSolver.store_results_2!(g[order], rhs1, ws.nstate, ws.nshock, order)
        @test g[2][:, [3, 8, 13, 18, 4, 9, 14, 19]] == rhs1
        
        @test results_perturbation_ws.g[2][:, [3, 4, 8, 9]] ≈ 2*r_ug2_3[:,[3, 4, 7, 8]]
        @test results_perturbation_ws.g[2][:, vcat(1:4, 6:9, 11:14, 16:19)] ≈ 2*r_ug2_3
    end

    @testset "σ²" begin
        KOrderSolver.update_gg_1!(ws.gg, g, 2, ws)
        @test ws.gg[2][1:2, vcat(1:4, 8:11, 15:18, 22:25)] == g[2][ws.state_index, vcat(1:4, 6:9, 11:14, 16:19)]

        KOrderSolver.make_hh!(ws.hh, g, ws.gg, 2, ws)
        KOrderSolver.update_hh!(ws.hh, g, ws.gg, 2, ws)
        @test ws.hh[2][3:5, vcat(1:4, 8:11, 15:18, 22:25)]  == g[2][:, vcat(1:4, 6:9, 11:14, 16:19)]
        target = g[2][1:2, vcat(1:2, 6:7)]*kron(g[1][2:3, 1:4], g[1][2:3, 1:4]) + g[1][1:2, 1:2]*g[2][2:3, vcat(1:4, 6:9, 11:14, 16:19)]
        @test ws.hh[2][6:7, vcat(1:4, 8:11, 15:18, 22:25)] ≈ target
#        gyuσΣ = zeros(ws.nvar, (ws.nstate + 2*ws.nshock + 1)^2)
#        KOrderSolver.collect_future_shocks!(gyuσΣ, g[2], 1, 1, 1, ws.nstate, ws.nvar, ws.nshock)
#        display(ws.g[2])
#        display(ws.ws.gyuσΣ)
    end
end

order = 3
#k_order_ws_3 = KOrderWs(endo_nbr,length(ifwd),length(ipre),endo_nbr,exo_nbr,ifwd,ipre,collect(1:endo_nbr),1:length(ipre),order)
#k_order_solution!(results_perturbation_ws.g, f, moments, order, ws)


end
