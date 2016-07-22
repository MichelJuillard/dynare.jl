include("first_order_solver.jl")
using Base.Test
using MAT

lli = [0     0     1     2     0     3;
       4     5     6     7     8     9;
       10    11     0     0     0    12]

n = 100
lli2 = repmat(lli,1,n)'
lli2 = lli2[:]
k = find(lli2')
lli2[k] = 1:length(k)
lli2 = reshape(lli2,n*6,3)'

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
jacobian2 = kron(eye(n),jacobian[:,1:12])

m = Model(6*n,lli2)


function solver_loop(iter,algo,jacobian,model,options)
    for i=1:iter
        ghx,gx,hx = first_order_solver(algo,jacobian,model,options)
    end
end

ghx,gx,hx = first_order_solver("GS", jacobian2, m, options)
@time ghx,gx,hx = first_order_solver("GS", jacobian2, m, options)
solver_loop(10,"GS",jacobian2,m,options)
@time solver_loop(10,"GS",jacobian2,m,options)
