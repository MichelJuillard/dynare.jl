using Base.Test

include("discrete_lyapunov.jl")

srand(123)
ws = DiscreteLyapunovWS(randn(3,3),3,3)
t=QuasiUpperTriangular([randn(2,3); 0 0 1])
b = randn(2,3)
b_orig = copy(b)
solve_two_rows!(ws.linsolve_ws,2,3,t,b,ws.w2,ws.b2,ws.b2n)
@test b - t[1:2,1:2]*b*t.' ≈ b_orig

n = 7
a = randn(n,n)
S = schur(a)
t = QuasiUpperTriangular(S[1])
b = randn(n,n)
b0 = copy(b)
b1 = copy(b)
ws = DiscreteLyapunovWS(a,n,n)

solve_one_row!(n,n,t,b,ws.w1,ws.b1)

ttarget = eye(n) - t[n,n]*t.'
xtarget = copy(b0)
xtarget[n,:] = view(b0,n:n,:)*inv(ttarget)
xtarget[1:n-1,:] += t[1:n-1,n].*xtarget[n:n,:]*t.'
@test xtarget ≈ b

solve_two_rows!(ws.linsolve_ws,2,n,t,b1,ws.w2,ws.b2,ws.b2n)
@test b1[1:2,:] - t[1:2,1:2]*b1[1:2,:]*t.' ≈ b0[1:2,:]

a = randn(n,n)
aorig = copy(a)
b = randn(n,n)
x = similar(b)
b0 = copy(b)
discrete_lyapunov_solver!(ws,a,b,x)

@test x - aorig*x*aorig' ≈ b0

bb = b*b'
a = copy(aorig)
bb0 = copy(bb)
bb1 = copy(bb)
x1 = similar(x)
x0 = reshape((eye(n*n) - kron(a,a))\vec(bb),n,n)
discrete_lyapunov_solver!(ws,a,bb,x)
@test x - aorig*x*aorig' ≈ bb0
@test x ≈ x0

a = copy(aorig)
discrete_lyapunov_symmetrical_solver!(ws,a,bb1,x1)
@test x1 - aorig*x1*aorig' ≈ bb0
