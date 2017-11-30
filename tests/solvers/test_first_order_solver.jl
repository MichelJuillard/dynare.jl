push!(LOAD_PATH,"../../src/models/")

include("../../src/dynare.jl")
using .Dynare

import .Dynare.DynLinAlg.Kronecker.a_mul_kron_b!

include("make_model.jl")

using Base.Test
using MAT
using Dynare.FirstOrder: FirstOrderSolverWS, remove_static!, get_de, first_order_solver

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
        
file = matopen("example1_results.mat")
oo_ = read(file,"oo_")

function test_model(endo_nbr,lead_lag_incidence)
    m = Model(endo_nbr,lead_lag_incidence, 2, 0)

    @test m.DErows1 == 1:5
    @test m.DErows2 == 6:6
    @test m.n_dyn == 6
    @test m.icolsD == [1, 2, 4, 5, 6]
    @test m.jcolsD == [6, 7, 10, 11, 12]
    @test m.icolsE == [1, 2, 3, 4, 5, 6]
    @test m.jcolsE == [1, 2, 3, 4, 5, 9]
    @test m.colsUD == 3:3
    @test m.colsUE == 6:6
end

function test_getDE(endo_nbr,lead_lag_incidence,jacobian)
    m = Model(endo_nbr,lead_lag_incidence, 2, 0)
    ws = FirstOrderSolverWS("GS", jacobian, m)
    remove_static!(ws,jacobian,m.p_static)
    @test norm(jacobian[m.n_static+1:end,m.p_static] - zeros(m.endo_nbr-m.n_static,m.n_static),Inf) < 1e-15
    D,E = get_de(jacobian[m.n_static+1:end,:],m)
    Dtarget = zeros(6,6)
    Dtarget[1:5,[1, 2, 4, 5, 6]] = jacobian[2:6,[6, 7, 10, 11, 12]]
    Dtarget[6,3] = 1
    Etarget = -[jacobian[2:6,[1, 2, 3, 4, 5, 9]]; [0 0 0 0 0 -1]] 
    @test D == Dtarget
    @test E == Etarget 
end

function test_solver(endo_nbr,lead_lag_incidence,options,algo,jacobian)
    m = Model(endo_nbr,lead_lag_incidence, 2, 0)
    ws = FirstOrderSolverWS(algo, jacobian, m)
    results = ResultsPerturbationWs(m, 1)
    remove_static!(ws,jacobian,m.p_static)
    @test size(jacobian) == (6,14)
    println("small model first round")
    first_order_solver(results, ws,algo, jacobian, m, options)
    println("small model second round")
    @time first_order_solver(results, ws,algo, jacobian, m, options)
    display(results.g[1])
    println("")
    display(oo_["dr"]["ghx"][squeeze(round.(Int,oo_["dr"]["inv_order_var"]),2),:])
    println("")
    display(oo_["dr"]["ghu"][squeeze(round.(Int,oo_["dr"]["inv_order_var"]),2),:])
    println("")
    res = norm(results.g[1][:,1:(m.n_bkwrd + m.n_both)]-oo_["dr"]["ghx"][squeeze(round.(Int,oo_["dr"]["inv_order_var"]),2),:],Inf)
    @test res < 1e-13
end

function solve_large_model(endo_nbr,lead_lag_incidence,options,algo,jacobian)
    m = Model(endo_nbr,lead_lag_incidence, 2, 0)
    ws = FirstOrderSolverWS(algo, jacobian, m)
    results = ResultsPerturbationWs(m, 1)
    @time    first_order_solver(results, ws,algo, jacobian, m, options)
end    

lli, jacobian = make_model(1)
test_model(6,lli)
test_getDE(6, lli, jacobian)
jacobian = hcat(jacobian,[ 0 0; 0 0; 0 0; 0 0; -1 0; 0 -1])
test_solver(6, lli, options, "CR", jacobian)
#test_solver(6, lli, options, "GS", jacobian)

n = 100
lli2, jacobian2 = make_model(n)
fu = zeros(6*n,2*n)
col = 1
row = 5
for i=1:n
    fu[row,col] = 1
    fu[row+1, col+1] = 1
    row += 6
    col += 2
end
jacobian2 = hcat(jacobian2, fu)
println("large model")
solve_large_model(n*6,lli2,options,"CR",jacobian2)
solve_large_model(n*6,lli2,options,"CR",jacobian2)
println("OK")
#solve_large_model(n*6,lli2,options,"GS",jacobian2)
#solve_large_model(n*6,lli2,options,"GS",jacobian2)



