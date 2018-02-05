using Base.Test
push!(LOAD_PATH, "../../src/linalg")
using ExtendedMul


m = 3
n = 4
ka = 2
kb = 3
a = randn(m*n + ka - 1)
b = randn(m*n + kb - 1)

kc = 4
c = randn(m*m + kc - 1)

A_mul_B!(c, kc, a, ka, m, n, b, kb, m)
@test reshape(c[kc:end], m, m) ≈ reshape(a[ka:end], m, n) * reshape(b[kb:end], n, m)

a = randn(3,4)
a1 = view(a,:,2:4)
ka1 = 1
ma = 3
na = 3
mb = 3
nb = 4

kc = 4
c = randn(ma*nb + kc - 1)

A_mul_B!(c, kc, a1, ka1, ma, na, b, kb, nb)
@test reshape(c[kc:kc+ma*nb-1], ma, nb) ≈ a1 * reshape(b[kb:end], mb, nb)
@test_throws DimensionMismatch A_mul_B!(c, kc, a1, 2, ma, na, b, kb, nb)


ka = 2
kb = 1
ma = 3
na = 4
mb = 4
nb = 3
a = randn(ma*na + ka - 1)
b = randn(mb,nb+1)
b1 = view(b,:,2:nb+1)

kc = 4
c = randn(ma*nb + kc - 1)

A_mul_B!(c, kc, a, ka, ma, na, b1, kb, nb)
@test reshape(c[kc:kc+ma*nb-1], ma, nb) ≈ reshape(a[ka:end],ma,na) * b1
@test_throws DimensionMismatch A_mul_B!(c, kc, a, ka, ma, na, b1, 2, nb)

m = 3
n = 4
ka = 2
kb = 3
a = randn(m*n + ka - 1)
b = randn(m*n + kb - 1)

kc = 4
c = randn(n*n + kc - 1)

At_mul_B!(c, kc, a, ka, n, m, b, kb, n)
@test reshape(c[kc:end], n, n) ≈ reshape(a[ka:end], m, n)' * reshape(b[kb:end], m, n)

kc = 4
c = randn(m*m + kc - 1)

A_mul_Bt!(c, kc, a, ka, m, n, b, kb, m)
@test reshape(c[kc:end], m, m) ≈ reshape(a[ka:end], m, n) * reshape(b[kb:end], m, n)'
