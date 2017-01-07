include("make_model.jl")
include("first_order_solver.jl")
using Base.Test
using MAT
using FirstOrder: Model, FirstOrderSolverWS, remove_static, get_DE, first_order_solver, add_static, QrWS, GsSolverWS, CycleReductionWS, cycle_reduction_core, gs_solver_core!

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


function solver_loop(iter,ws,algo,jacobian,model,options)
    for i=1:iter
        ghx,gx,hx = first_order_solver(ws,algo,jacobian,model,options)
    end
end

function timing_test(n,algo)
    lli2, jacobian2 = make_model(n)
    m = Model(6*n,lli2)
    ws = FirstOrderSolverWS(algo, jacobian2, m)
    ghx,gx,hx = first_order_solver(ws,algo, jacobian2, m, options)
    @time ghx,gx,hx = first_order_solver(ws,algo, jacobian2, m, options)
    solver_loop(10,ws,algo,jacobian2,m,options)
    @time solver_loop(10,ws,algo,jacobian2,m,options)
end

for algo in ["CR"]
    for n in [1, 100]
        timing_test(n,algo)
    end
end
