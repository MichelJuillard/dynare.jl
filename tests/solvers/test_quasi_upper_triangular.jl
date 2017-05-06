using Base.Test

include("quasi_upper_triangular.jl")

Aorig = [1 3; 0.5 2]
A = QuasiUpperTriangular(Aorig)
B = eye(2)
X = zeros(2,2)
A_ldiv_B!(A,B)

@test B ≈ inv(Aorig)

B = eye(2)
A_rdiv_Bt!(B,A)
@test B ≈ inv(Aorig).'

B = eye(2)
A_rdiv_B!(B,A)

@test B ≈ inv(Aorig)

srand(123)
n = 7
a = randn(n,n)
S = schur(a)
t = S[1]
b = randn(n,n)
c = similar(b)

@test t*b ≈ A_mul_B!(QuasiUpperTriangular(t),b)
@test t'*b ≈ At_mul_B!(QuasiUpperTriangular(t),b)
@test b*t ≈ A_mul_B!(b,QuasiUpperTriangular(t))
@test b*t' ≈ A_mul_Bt!(b,QuasiUpperTriangular(t))

@test t*b ≈ A_mul_B!(c,QuasiUpperTriangular(t),b)
@test t'*b ≈ At_mul_B!(c,QuasiUpperTriangular(t),b)
@test b*t ≈ A_mul_B!(c,b,QuasiUpperTriangular(t))
@test b*t' ≈ A_mul_Bt!(c,b,QuasiUpperTriangular(t))

b1 = copy(b)
x = zeros(n,n)
A_ldiv_B!(QuasiUpperTriangular(t),b1)
@test t\b ≈ b1
b1 = copy(b)
A_rdiv_B!(b1,QuasiUpperTriangular(t))
@test b/t ≈ b1
b1 = copy(b)
A_rdiv_Bt!(b1,QuasiUpperTriangular(t))
@test b/t.' ≈ b1

b = rand(n)
b1 = copy(b)
r = rand()
I_plus_rA_ldiv_B!(r,QuasiUpperTriangular(t),b1)
@test b1 ≈ (eye(n) + r*t)\b
b1 = copy(b)
s = rand()
I_plus_rA_plus_sB_ldiv_C!(r,s,QuasiUpperTriangular(t),QuasiUpperTriangular(t*t),b1)
@test b1 ≈ (eye(n) + r*t + s*t*t)\b

b = rand(n,n)
b1 = copy(b)
r = rand()
I_plus_rA_ldiv_B!(r,QuasiUpperTriangular(t),b1)
@test b1 ≈ (eye(n) + r*t)\b
b1 = copy(b)
s = rand()
I_plus_rA_plus_sB_ldiv_C!(r,s,QuasiUpperTriangular(t),QuasiUpperTriangular(t*t),b1)
@test b1 ≈ (eye(n) + r*t + s*t*t)\b

