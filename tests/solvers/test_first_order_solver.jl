include("first_order_solver.jl")
using Base.Test
using MAT

lli = [0     0     1     2     0     3;
       4     5     6     7     8     9;
       10    11     0     0     0    12]

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

file = matopen("jacobian.mat")
jacobian = read(file,"jacobia")
file = matopen("example1_results.mat")
oo_ = read(file,"oo_")

function test_model(endo_nbr,lead_lag_incidence)
    m = Model(endo_nbr,lead_lag_incidence)

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
    m = Model(endo_nbr,lead_lag_incidence)
    Q, jacobian_ = remove_static(jacobian,m.i_static)
    D,E = get_DE(jacobian_[m.n_static+1:end,:],m)
    Dtarget = zeros(6,6)
    Dtarget[1:5,[1, 2, 4, 5, 6]] = jacobian_[:,[6, 7, 10, 11, 12]]
    Dtarget[6,3] = 1
    Etarget = -[jacobian_[:,[1, 2, 3, 4, 5, 9]]; [0 0 0 0 0 -1]] 
    @test D == Dtarget
    @test E == Etarget 
end

function test_solver(endo_nbr,lead_lag_incidence,options,algo,jacobian)
    m = Model(endo_nbr,lead_lag_incidence)
    Q, jacobian_ = remove_static(jacobian,m.p_static)
    @test size(Q) == (6,6)
    @test size(jacobian_) == (6,14)
    D,E = get_DE(jacobian_[m.n_static+1:end,:],m)
    ghx,gx,hx = first_order_solver(algo, jacobian, m, options)
    res = norm(ghx-oo_["dr"]["ghx"][squeeze(round(Int,oo_["dr"]["inv_order_var"]),2),:],Inf)
    @test res < 1e-13
end

#test_model(6,lli)
#test_getDE(6, lli, jacobian)
test_solver(6, lli, options, "GS", jacobian)
