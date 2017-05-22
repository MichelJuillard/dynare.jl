include("solve_eye_plus_at_kron_b_optim.jl")

using Base.Test
using MAT

n = 3
x = randn(n,n*n)
a = randn(n,n)
b = randn(n,n)
c = randn(n,n)
d = a*x + b*x*kron(c,c)
dold = d
ws = EyePlusAtKronBWS(a,b,2,c)

index = 1
depth = 1
nd = n^depth
t = QuasiUpperTriangular(lu(a\b)[2])
s = QuasiUpperTriangular(lu(c)[2])
d = zeros(n^2)
r = 1.0
d[1:n] = randn(n)
y = copy(d[1:n])
solvi_real_eliminate!(index,n,nd,1:nd,depth-1,r,t,s,d,ws)
d_target = -kron(s[1,:],t)*y
@test d[n+1:n^2] ≈ d_target[n+1:n^2]

index = 1
depth = 3
ws = EyePlusAtKronBWS(a,b,depth,c)
nd = n^depth
t = QuasiUpperTriangular(lu(a\b)[2])
s = QuasiUpperTriangular(lu(c)[2])
d = zeros(n^(depth+1))
r = 1.0
d[1:nd] = randn(nd)
y = copy(d[1:nd])
solvi_real_eliminate!(index,n,nd,1:nd,depth-1,r,t,s,d,ws)
d_target = -kron(s[1,:],kron(kron(s',s'),t))*y
@test d[nd+1:n^(depth+1)] ≈ d_target[nd+1:n^(depth+1)]

index = 2
d = zeros(n^(depth+1))
r = 1.0
d[nd+(1:nd)] = randn(nd)
y = copy(d[nd+(1:nd)])
solvi_real_eliminate!(index,n,nd,nd+(1:nd),depth-1,r,t,s,d,ws)
d_target = -kron(s[2,:],kron(s',s'),t)*y
@test d[2*nd+1:n^(depth+1)] ≈ d_target[2*nd+1:n^(depth+1)]

aa = a\b
t = QuasiUpperTriangular(lu(aa)[2])
s = QuasiUpperTriangular(lu(c)[2])

r = 1.0
d = randn(n)
d_orig = copy(d)
td = similar(d)
order = 0
depth = order
t2 = t*t
s2 = s*s
solve1!(r,depth,t,t2,s,s2,d,ws)
d_target = (eye(n) + t)\d_orig
@test d_target ≈ d

r = 1.0
d = randn(n*n*n)
d_orig = copy(d)
td = similar(d)
order = 2
depth = order
solve1!(r,depth,t,t2,s,s2,d,ws)

d_target = (eye(n^3) + kron(kron(s',s'),t))\d_orig

@test d ≈ d_target

a = 0.5
b1 = 0.1
b2 = 1.3
depth = 1
tt = QuasiUpperTriangular(t)
ss = QuasiUpperTriangular(s)
d = randn(2*n^(depth+1))
d_orig = copy(d)
transformation1(a,b1,b2,depth,tt,ss,d,ws)
d_target = d_orig + kron([a -b1; -b2 a],kron(s',t))*d_orig

@test d ≈ d_target

index = 1
depth = 3
nd = n^depth
t = QuasiUpperTriangular(lu(a\b)[2])
s = QuasiUpperTriangular(lu(c)[2])
s[1,2] = 0.5
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
d = zeros(n^(depth+1))
d[1:nd] = randn(nd)
d_orig = copy(d)
r1 = 1.0
r2 = 0.8
td = []
solviip_real_eliminate!(index,n,nd,1:nd,depth-1,r1,r2,t,t2,s,s2,d,ws)
d_target = -2*r1*kron(s[index,:],kron(kron(s',s'),t))*d_orig[1:nd] - (r1*r1+r2*r2)*kron(s2[index,:],kron(kron(s2',s2'),t2))*d_orig[1:nd]
@test d[nd+1:n^(depth+1)] ≈ d_target[nd+1:n^(depth+1)]
           
depth = 2
nd = n^depth
gamma = 0.3
delta1 = -0.4
delta2 = 0.6
d = randn(2*nd)
d_orig = copy(d)
transform2(r1, r2, gamma, delta1, delta2, nd, depth, t, t2, s, s2, d, ws)
G = [gamma delta1; delta2 gamma]
d_target = (eye(2*nd) + 2*r1*kron(G,kron(s',t)) + (r1*r1 + r2*r2)*kron(G*G,kron(s2',t2)))*d_orig
@test d ≈ d_target

index = 1
depth = 3
nd = n^depth
t = QuasiUpperTriangular(schur(a\b)[1])
s = QuasiUpperTriangular(schur(c)[1])
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
d = zeros(n^(depth+1))
d[1:2*nd] = randn(2*nd)
d_orig = copy(d)
r1 = 1.0
r2 = 0.8
td = []
solviip_complex_eliminate!(index,n,nd,1:nd,depth-1,r1,r2,t,t2,s,s2,d,ws)
d_target = (-2*r1*kron(kron(s[1,:],kron(s',s')),t)*d_orig[1:nd] - (r1*r1+r2*r2)*kron(kron(s2[1,:],kron(s2',s2')),t2)*d_orig[1:nd]
            -2*r1*kron(kron(s[2,:],kron(s',s')),t)*d_orig[nd+1:2*nd] - (r1*r1+r2*r2)*kron(kron(s2[2,:],kron(s2',s2')),t2)*d_orig[nd+1:2*nd])
@test d[2*nd+1:n^3] ≈ d_target[2*nd+1:n^3]

depth = 1
n = 2
nd = n^depth
t = QuasiUpperTriangular(schur([1 2; -5 4])[1])
s = QuasiUpperTriangular(schur([1 -3; 2 3])[1])
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
d = randn(n^(depth+1))
d_orig = copy(d)
alpha = s[1,1]
# s is transposed
beta1 = s[2,1]
beta2 = s[1,2]
beta = sqrt(-beta1*beta2)
td = similar(d)
solviip(alpha,beta,depth,t,t2,s,s2,d,ws)
d_target = (eye(n^2) + 2*alpha*kron(s',t) + (alpha*alpha + beta*beta)*kron(s2',t2))\d_orig 
@test d ≈ d_target

depth = 1
n = 3
#srand(1) #first diag block 2x2
#srand(2) # upper triangular
#srand(3) # second diag block 2x2
a = randn(n,n)
b = randn(n,n)
c = randn(n,n)
ws = EyePlusAtKronBWS(a,b,2,c)
nd = n^depth
t = QuasiUpperTriangular(schur(a\b)[1])
s = QuasiUpperTriangular(schur(c)[1])
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
d = randn(n^(depth+1))
d_orig = copy(d)
alpha = s[1,1]
# s is transposed
beta1 = s[2,1]
beta2 = s[1,2]
r1 = alpha
r2 = sqrt(-beta1*beta2)
td = similar(d)
println("begin test")
solviip(r1,r2,depth,t,t2,s,s2,d,ws)
d_target = (eye(n^2) + 2*alpha*kron(s',t) + (alpha*alpha + r2*r2)*kron(s2',t2))\d_orig 
@test d ≈ d_target

depth = 2
nd = n^depth
t = QuasiUpperTriangular(schur(a\b)[1])
s = QuasiUpperTriangular(schur(c)[1])
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
d = randn(n^3)
d_orig = copy(d)
alpha = s[1,1]
# s is transposed
beta1 = s[2,1]
beta2 = s[1,2]
r1 = alpha
r2 = sqrt(-beta1*beta2)
td = similar(d)
solviip(r1,r2,depth,t,t2,s,s2,d,ws)
d_target = (eye(n^3) + 2*alpha*kron(s',kron(s',t)) + (alpha*alpha + r2*r2)*kron(kron(s2',s2'),t2))\d_orig 
@test d ≈ d_target


n = 4
a = randn(n,n)
b = randn(n,n)
c = randn(n,n)
t = QuasiUpperTriangular(schur(a\b)[1])
s = QuasiUpperTriangular(schur(c)[1])
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
order = 3
d = randn(n^(order+1))
d_orig = copy(d)
ws = EyePlusAtKronBWS(a,b,order,c)
solver!(t,s,d,order,ws)
d_target = (eye(n^(order+1)) + kron(s',kron(kron(s',s'),t)))\d_orig
@test d ≈ d_target

n = 4
a = randn(n,n)
b = randn(n,n)
c = randn(n,n)
t = QuasiUpperTriangular(schur(a\b)[1])
s = QuasiUpperTriangular(schur(c)[1])
s2 = QuasiUpperTriangular(s*s)
t2 = QuasiUpperTriangular(t*t)
order = 3
ws = EyePlusAtKronBWS(a,b,order,c)
d = randn(n^(order+1))
d_orig = copy(d)
solver!(t,s,d,order,ws)
d_target = (eye(n^(order+1)) + kron(s',kron(kron(s',s'),t)))\d_orig
@test d ≈ d_target


n1 = 4
n2 = 3
a = randn(n1,n1)
b = randn(n1,n1)
c = randn(n2,n2)
order = 2
ws = EyePlusAtKronBWS(a,b,order,c)
d = randn(n1,n2^order)
a_orig = copy(a)
b_orig = copy(b)
c_orig = copy(c)
d_orig = copy(d)

general_sylvester_solver!(a,b,c,d,2,ws)
@test a_orig*d + b_orig*d*kron(c_orig,c_orig) ≈ d_orig
@test d ≈ reshape((kron(eye(n2^order),a_orig) + kron(kron(c_orig',c_orig'),b_orig))\vec(d_orig),n1,n2^order)

function f(t,s,d,order,ws)
    for i = 1:100
        solver!(t,s,d,order,ws)
    end
end
    
#@profile  f(t,s,d,order,ws)
#Profile.print(combine=true,sortedby=:count)
