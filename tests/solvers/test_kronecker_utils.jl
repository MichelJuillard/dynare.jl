using Base.Test
include("quasi_upper_triangular.jl")
include("kronecker_utils.jl")

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
                @time kron_mul_elem!(p, q, m, a, d, w)
                @test d ≈ kron(kron(eye(n^p),a),eye(m*n^q))*d_orig
                d = copy(d_orig)
                @time kron_mul_elem!(p, q, m, t, d, w)          
                @test d ≈ kron(kron(eye(n^p),t),eye(m*n^q))*d_orig
                d = copy(d_orig)
                @time kron_mul_elem_t!(p, q, m, a, d, w)
                @test d ≈ kron(kron(eye(n^p),a'),eye(m*n^q))*d_orig
                d = copy(d_orig)
                @time kron_mul_elem_t!(p, q, m, t, d, w)          
                @test d ≈ kron(kron(eye(n^p),t'),eye(m*n^q))*d_orig
            end
        end
    end
end

ma = 2
na = 3
a = randn(ma,na)
mc = 3
order = 4
c = randn(mc,mc)
b = randn(na,mc^order)
b_orig = copy(b)
d = Matrix{Float64}(ma,mc^order)
w = Vector{Float64}(ma*mc^order)
a_mul_b_kron_c!(d,a,b,c,order,w)
cc = c
for i = 2:order
    cc = kron(cc,c)
end
@test d ≈ a*b_orig*cc

w = Vector{Float64}(na*mc^order)
b = copy(b_orig)
a_mul_kron_b!(b,c,order,w)
cc = c
for i = 2:order
    cc = kron(cc,c)
end
@test b ≈ b_orig*cc

