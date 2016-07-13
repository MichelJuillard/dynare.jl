include("cycle_reduction.jl")

using CycleReduction

# Set the dimension of the problem to be solved.
n = 500

# Set the equation to be solved
A = eye(n)
B = diagm(30*ones(n))
B[1,1] = 20
B[end,end] = 20
B = B - diagm(10*ones(n-1),-1)
B = B - diagm(10*ones(n-1),1)
C = diagm(15*ones(n))
C = C - diagm(5*ones(n-1),-1)
C = C - diagm(5*ones(n-1),1)

cycle_reduction(C,B,A,1e-7,true)
@time cycle_reduction(C,B,A,1e-7)
n,m = size(C)
id0, id2, A0_0, Ahat1, tmp = cycle_reduction_init(n)
@time info = cycle_reduction_core(C, B, A, 1e-7, 300, A0_0, Ahat1, tmp, id0, id2)

