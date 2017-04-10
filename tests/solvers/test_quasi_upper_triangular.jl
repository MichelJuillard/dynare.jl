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
a = randn(7,7)
S = schur(a)
t = S[1]
b = randn(7,7)

@test t*b ≈ A_mul_B!(QuasiUpperTriangular(t),b)
@test t'*b ≈ At_mul_B!(QuasiUpperTriangular(t),b)
@test b*t ≈ A_mul_B!(b,QuasiUpperTriangular(t))
@test b*t' ≈ A_mul_Bt!(b,QuasiUpperTriangular(t))

b1 = copy(b)
x = zeros(7,7)
A_ldiv_B!(QuasiUpperTriangular(t),b1)
@test t\b ≈ b1
b1 = copy(b)
A_rdiv_B!(b1,QuasiUpperTriangular(t))
@test b/t ≈ b1
b1 = copy(b)
A_rdiv_Bt!(b1,QuasiUpperTriangular(t))
@test b/t.' ≈ b1
