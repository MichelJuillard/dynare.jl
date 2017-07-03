using Base.Test
include("../../src/linalg/quasi_upper_triangular.jl")
using QUT
include("../../src/linalg/kronecker_utils.jl")
using Kronecker

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
                @time Kronecker.kron_mul_elem!(w, a, d, p, q, m)
                @test w ≈ kron(kron(eye(n^p),a),eye(m*n^q))*d_orig
                d = copy(d_orig)
                @time Kronecker.kron_mul_elem!(w, t, d, p, q, m)          
                @test w ≈ kron(kron(eye(n^p),t),eye(m*n^q))*d_orig
                d = copy(d_orig)
                @time Kronecker.kron_mul_elem_t!(w, a, d, p, q, m)
                @test w ≈ kron(kron(eye(n^p),a'),eye(m*n^q))*d_orig
                d = copy(d_orig)
                @time Kronecker.kron_mul_elem_t!(w, t, d, p, q, m)          
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
d = Matrix{Float64}(ma,mc^order)
w = Vector{Float64}(ma*mc^order)
Kronecker.a_mul_b_kron_c!(d,a,b,c,order)
cc = c
for i = 2:order
    cc = kron(cc,c)
end
@test d ≈ a*b_orig*cc

b = copy(b_orig)
Kronecker.a_mul_kron_b!(d,b,c,order)
cc = c
for i = 2:order
    cc = kron(cc,c)
end
@test d ≈ b_orig*cc

