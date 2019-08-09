using Test
using LinearAlgebra
using Random

push!(LOAD_PATH, "../src/linalg")
using QUT

Aorig = [1 3; 0 2]
A1 = [Aorig Aorig; zeros(2,2) Aorig]
A = QuasiUpperTriangular(A1)
B = randn(4,4)
C = zeros(4,4)
mul!(C, 1, B, 1, 4, 4, A, 1, 4)

@test C ≈ B*A1

Aorig = [1 3; 0.5 2]
A = QuasiUpperTriangular(Aorig)
B = Matrix(1.0I, 2, 2)
X = zeros(2,2)
ldiv!(A,B)

@test B ≈ inv(Aorig)

B = Matrix(1.0I, 2, 2)
rdiv!(B,transpose(A))
@test B ≈ transpose(inv(Aorig))

B = Matrix(1.0I, 2, 2)
rdiv!(B,A)

@test B ≈ inv(Aorig)

n = 7
a = randn(n,n)
S = schur(a)
t = S.Schur
b = randn(n,n)
c = similar(b)

@test t*b ≈ mul!(QuasiUpperTriangular(t),b)
@test transpose(t)*b ≈ mul!(transpose(QuasiUpperTriangular(t)),b)
@test b*t ≈ mul!(b,QuasiUpperTriangular(t))
@test b*transpose(t) ≈ mul!(b,transpose(QuasiUpperTriangular(t)))

@test t*b ≈ mul!(c,QuasiUpperTriangular(t),b)
@test transpose(t)*b ≈ mul!(c,transpose(QuasiUpperTriangular(t)),b)
@test b*t ≈ mul!(c,b,QuasiUpperTriangular(t))
@test b*transpose(t) ≈ mul!(c,b,transpose(QuasiUpperTriangular(t)))

b1 = copy(b)
x = zeros(n,n)
ldiv!(QuasiUpperTriangular(t),b1)
@test t\b ≈ b1
b1 = copy(b)
rdiv!(b1,QuasiUpperTriangular(t))
@test b/t ≈ b1
b1 = copy(b)
rdiv!(b1,transpose(QuasiUpperTriangular(t)))
@test b/transpose(t) ≈ b1

b = rand(n)
b1 = copy(b)
r = rand()
I_plus_rA_ldiv_B!(r,QuasiUpperTriangular(t),b1)
@test b1 ≈ (Matrix(1.0I, n, n) + r*t)\b
b1 = copy(b)
s = rand()
I_plus_rA_plus_sB_ldiv_C!(r,s,QuasiUpperTriangular(t),QuasiUpperTriangular(t*t),b1)
@test b1 ≈ (Matrix(1.0I, n, n) + r*t + s*t*t)\b

b = rand(n,n)
b1 = copy(b)
r = rand()
I_plus_rA_ldiv_B!(r,QuasiUpperTriangular(t),b1)
@test b1 ≈ (Matrix(1.0I, n, n) + r*t)\b
b1 = copy(b)
s = rand()
I_plus_rA_plus_sB_ldiv_C!(r,s,QuasiUpperTriangular(t),QuasiUpperTriangular(t*t),b1)
@test b1 ≈ (Matrix(1.0I, n, n) + r*t + s*t*t)\b

