import Base.BLAS: gemm!
using Base.Test
push!(LOAD_PATH, "../../src/linalg")
using QUT
using KroneckerUtils

srand(123)
a = rand(2,3)
b = rand(3,4)
c = zeros(8)

gemm!('N', 'N', 1.0, vec(a), 2, 3, vec(b), 4, 1.0, c)
@test reshape(c,2,4) ≈ a*b

for m in [1, 3]
    for n in [3, 4]
        a = randn(n,n)
        t = QuasiUpperTriangular(schur(a)[1])
        depth = 4
        for p = 0:4
            for q = depth - p
                d = randn(m*n^(p+q+1))
                d_orig = copy(d)
                w = similar(d)
                println("m = $m, p = $p, q = $q")
                d = copy(d_orig)
                @time KroneckerUtils.kron_mul_elem_t!(w, a, d, n^p, n^q*m)
                @test w ≈ kron(kron(eye(n^p),a'),eye(m*n^q))*d_orig
                d = copy(d_orig)
                @time KroneckerUtils.kron_mul_elem_t!(w, t, d, n^p, n^q*m)
                @test w ≈ kron(kron(eye(n^p),t'),eye(m*n^q))*d_orig
            end
        end
    end
end

ma = 2
na = 2
a = randn(ma,na)
mc = 3
order = 4
c = randn(mc,mc)
b = randn(na,mc^order)
b_orig = copy(b)
d = randn(ma,mc^order)
w1 = Vector{Float64}(ma*mc^order)
w2 = Vector{Float64}(ma*mc^order)
KroneckerUtils.a_mul_b_kron_c!(d, a, b, c, order, w1, w2)
cc = c
for i = 2:order
    cc = kron(cc,c)
end
@test d ≈ a*b_orig*cc

b = copy(b_orig)
KroneckerUtils.a_mul_kron_b!(d,b,c,order)
cc = c
for i = 2:order
    cc = kron(cc,c)
end
@test d ≈ b_orig*cc

order = 3
ma = 2
na = 4
a = randn(ma,na)
mb1 = 2
nb1 = 4
b1 = randn(mb1,nb1)
mb2 = 2
nb2 = 2
b2 = randn(mb2,nb2)
b = [b1, b2]
c = randn(ma,nb1*nb2)
work = zeros(16)
KroneckerUtils.a_mul_kron_b!(c,a,b,work)
@test c ≈ a*kron(b[1],b[2])

order = 3
ma = 2
na = 4
a = randn(ma,na)
mb = 4
nb = 8
b = randn(mb,nb)
c = randn(2,2)
d = randn(2,2)
work1 = zeros(mb*nb)
work2 = zeros(mb*nb)
e = randn(ma,8)
KroneckerUtils.a_mul_b_kron_c_d!(e, a, b, c, d, order, work1, work2)
@test e ≈ a*b*kron(c,kron(d,d))

order = 2
ma = 2
na = 4
a = randn(ma,na)
mb = 4
nb = 8
b = randn(mb,nb)
c = randn(ma*ma*nb)
d = randn(na*na*mb)
work1 = rand(na*na*mb)
work2 = rand(na*na*mb)
@time KroneckerUtils.kron_at_kron_b_mul_c!(d, a, order, b, c, work1, work2)
@time KroneckerUtils.kron_at_kron_b_mul_c!(d, a, order, b, c, work1, work2)

@test d ≈ kron(kron(a',a'),b)*c

println("test1")
order = 2
ma = 2
na = 4
q = 2
a = rand(ma, na)
b = rand(q*na^order)
c = rand(q*ma^order)
work1 = rand(q*max(ma, na)^order)
work2 = similar(work1)
KroneckerUtils.kron_a_mul_b!(c, a, order, b, q, work1, work2)
@time KroneckerUtils.kron_a_mul_b!(c, a, order, b, q, work1, work2)
@test c ≈ kron(kron(a,a),eye(q))*b

println("test2")
b = rand(q*ma^order)
c = rand(q*na^order)
KroneckerUtils.kron_at_mul_b!(c, a, order, b, q, work1, work2)
@time KroneckerUtils.kron_at_mul_b!(c, a, order, b, q, work1, work2)
@test c ≈ kron(kron(a',a'),eye(q))*b

println("test3")
order = 2
mc = 2
nc = 3
c = rand(mc,nc)
ma = 2
na = 4
a = rand(ma,na)
nb = nc^order
b = rand(na,nb)
d = rand(ma,mc^order)
work1 = rand(ma*max(mc, nc)^order)
work2 = similar(work1)
KroneckerUtils.a_mul_b_kron_ct!(d, a, b, c, order, work1, work2)
@time KroneckerUtils.a_mul_b_kron_ct!(d, a, b, c, order, work1, work2)
@test d ≈ a*b*kron(c',c')

println("test4")
nb = mc^order
b = rand(ma,nb)
d = rand(na,nc^order)
work1 = rand(na*max(mc, nc)^order)
work2 = similar(work1)
KroneckerUtils.at_mul_b_kron_c!(d, a, b, c, order, work1, work2)
@time KroneckerUtils.at_mul_b_kron_c!(d, a, b, c, order, work1, work2)
@test d ≈ a'*b*kron(c,c)
