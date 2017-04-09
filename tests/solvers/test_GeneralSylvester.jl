include("GeneralSylvester.jl")

using GeneralSylvester
using Base.Test
using MAT

n = 3
x = randn(n,n*n)
a = randn(n,n)
b = randn(n,n)
c = randn(n,n)
d = a*x + b*x*kron(c,c)
dold = d

index = [n,n]
j = 1
depth = 2
y = a\b*ones(n)
t = lu(a\b)[2]
s = lu(c)[2]
d = zeros(n^(depth+1))
r = 1.0
real_eliminate!(index,1:n,n^3-n+1,depth,y,r,t,s,d)
kf = kron(s,s)
d_target = -kron(kf[:,size(kf,2)],y)
@test d[1:end-n] ≈ d_target[1:end-n]

index = [1,2]
d = zeros(n^(depth+1))
real_eliminate!(index,1:n,n+1,depth,y,r,t,s,d)
d_target = -kron(kf[:,2],y)
@test d[1:n] ≈ d_target[1:n]

aa = a\b
t = lu(aa)[2]
s = lu(c)[2]

r = 1.0
d = randn(n)
d_orig = copy(d)
order = 0
depth = order
index = []
index_j = 1:n
solvi(r,index,index_j,depth,t,s,d)
d_target = (eye(n) + t)\d_orig
@test d_target ≈ d

r = 1.0
d = randn(n*n*n)
d_orig = copy(d)
order = 2
depth = order
index = zeros(Int64,depth)
index_j = collect(n^3-n+(1:n))
println("index_j ",index_j)
solvi(r,index,index_j,depth,t,s,d)

d_target = (eye(n^3) + kron(kron(s,s),t))\d_orig

@test d ≈ d_target

a = 0.5
b1 = 0.1
b2 = 1.3
depth = 2
tt = GeneralSylvester.QuasiUpperTriangular(t)
ss = GeneralSylvester.QuasiUpperTriangular(s)
d = randn(2*n^depth)
d_orig = copy(d)
transformation1(a,b1,b2,depth,tt,ss,d)
d_target = d_orig + kron([a -b1; -b2 a],kron(s,t))*d_orig

@test d ≈ d_target


#general_sylvester_solver!(a,b,c,d,2)

#@test d ≈ dold
