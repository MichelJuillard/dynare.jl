using MAT

include("cycle_reduction.jl")
#using CycleReduction

file = matopen("jacobian.mat")
jacobian = read(file,"jacobia")

n = 6
cycle_reduction_ws = CycleReduction(n)
A,B,C = get_ABC!(cycle_reduction_ws,model,jacobian)

# Set the dimension of the problem to be solved.
#n = 2

# Set the equation to be solved
#A = eye(n)
#B = diagm(30*ones(n))
#B[1,1] = 20
#B[end,end] = 20
#B = B - diagm(10*ones(n-1),-1)
#B = B - diagm(10*ones(n-1),1)
#C = diagm(15*ones(n))
#C = C - diagm(5*ones(n-1),-1)
#C = C - diagm(5*ones(n-1),1)
#C = -(A+B)

#cycle_reduction(C,B,A,1e-7,true)
#@time cycle_reduction(C,B,A,1e-7)
n,m = size(C)
println(A)
println(B)
println(C)
cycle_reduction_ws = CycleReductionWS(n,C,B,A)
println(cycle_reduction_ws.A)
info = cycle_reduction_core(cycle_reduction_ws, 1e-7,300)
cycle_reduction_check(cycle_reduction_ws.A,C,B,A,1e-7)


