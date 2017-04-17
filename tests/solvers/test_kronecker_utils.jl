include("quasi_upper_triangular.jl")
include("kronecker_utils.jl")

n = 2
a = randn(n,n)
t = QuasiUpperTriangular(schur(a)[1])
b = randn(n,n)
s = QuasiUpperTriangular(schur(b)[1])

level = 0
depth = 0
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level!(depth,level,s,d)
@test d ≈ s*d_orig

level = 0
depth = 2
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level!(depth,level,s,d)
@test d ≈ kron(eye(n^depth),s)*d_orig
             
level = 2
depth = 0
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level!(depth,level,s,d)
@test d ≈ kron(s,eye(n^level))*d_orig
             
level = 1
depth = 2
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level!(depth,level,s,d)
@test d ≈ kron(kron(eye(n^depth),s),eye(n^level))*d_orig
             
level = 3
depth = 3
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level!(depth,level,s,d)
@test d ≈ kron(eye(n^depth),kron(s,eye(n^level)))*d_orig

depth = 3
d = randn(n^(depth+1))
d_orig = copy(d)
mult_kron!(s,t,d,depth)
@test d ≈ kron(t,kron(t,kron(t,s)))*d_orig

n=10
a = randn(n,n)
t = QuasiUpperTriangular(schur(a)[1])
b = randn(n,n)
s = QuasiUpperTriangular(schur(b)[1])
depth = 3
d = randn(n^(depth+1))
d_orig = copy(d)
mult_kron!(s,t,d,depth)
@test d ≈ kron(t,kron(t,kron(t,s)))*d_orig

n = 2
a = randn(n,n)
t = QuasiUpperTriangular(schur(a)[1])
b = randn(n,n)
s = QuasiUpperTriangular(schur(b)[1])

level = 0
depth = 0
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level_t!(depth,level,s,d)
@test d ≈ s'*d_orig

level = 0
depth = 2
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level_t!(depth,level,s,d)
@test d ≈ kron(eye(n^depth),s')*d_orig
             
level = 2
depth = 0
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level_t!(depth,level,s,d)
@test d ≈ kron(s',eye(n^level))*d_orig
             
level = 1
depth = 2
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level_t!(depth,level,s,d)
@test d ≈ kron(kron(eye(n^depth),s'),eye(n^level))*d_orig
             
level = 3
depth = 3
d = randn(n^(level+depth+1))
d_orig = copy(d)
mult_level_t!(depth,level,s,d)
@test d ≈ kron(eye(n^depth),kron(s',eye(n^level)))*d_orig

depth = 3
d = randn(n^(depth))
d_orig = copy(d)
mult_kron_transposed!(t,depth,d)
@test d ≈ kron(t',kron(t',t'))*d_orig

n=10
a = randn(n,n)
t = QuasiUpperTriangular(schur(a)[1])
b = randn(n,n)
s = QuasiUpperTriangular(schur(b)[1])
depth = 3
d = randn(n^(depth))
d_orig = copy(d)
mult_kron_transposed!(t,depth,d)
@test d ≈ kron(t',kron(t',t'))*d_orig





