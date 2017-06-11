using Base.Test
"""
    a_mul_kron_b!(a::AbstractVecOrMat, b::AbstractMatrix, order::Int64, w::AbstractVector)

Performs a*(b ⊗ b ⊗ ... ⊗ b). The solution is returned in matrix or vector a. order indicates the number of occurences of b

We use vec(a*(b ⊗ b ⊗ ... ⊗ b)) = (b' ⊗ b' ⊗ ... ⊗ b' \otimes I)vec(a)

"""
function a_mul_kron_b!(a::AbstractVecOrMat, b::AbstractMatrix, order::Int64, w::AbstractVector)
    ma,na = size(a)
    mb,nb = size(b)
    mb == nb || throw(DimensionMismatch("B must be a square matrix"))
    mb^order == na || throw(DimensionMismatch("The number of columns of a, $(size(a,2)), doesn't match the number of rows of b, $(size(b,1)), times order = $(order)"))

    avec = vec(a)
    for p=1:order
        kron_mul_elem_t!(p-1,order-p,ma,b,avec,w)
    end
end
    
"""
    a_mul_b_kron_c!(d::AbstractVecOrMat, a::AbstractVecOrMat, b::AbstractMatrix, c::AbstractMatrix, order::Int64, w::AbstractVector)

Performs a*B*(c ⊗ c ⊗ ... ⊗ c). The solution is returned in matrix or vector a. order indicates the number of occurences of b. c must be a square matrix

We use vec(a*b*(c ⊗ c ⊗ ... ⊗ c)) = (c' ⊗ c' ⊗ ... ⊗ c' ⊗ a)vec(b)

"""
function a_mul_b_kron_c!(d::AbstractVecOrMat, a::AbstractVecOrMat, b::AbstractMatrix, c::AbstractMatrix, order::Int64, w::AbstractVector)
    ma, na = size(a)
    mb, nb = size(b)
    mc, nc = size(c)
    md, nd = size(d)
    mc == nc || throw(DimensionMismatch("C must be a square matrix"))
    mc^order == nb || throw(DimensionMismatch("The number of columns of a, $(size(a,2)), doesn't match the number of rows of b, $(size(b,1)), times order = $(order)"))
    (ma == md && nc^order == nd) || throw(DimensionMismatch("Dimension mismatch for D: $(size(d)) while ($ma, $(nc^order)) was expected"))
    A_mul_B!(d,a,b)
    dvec = vec(d)
    for p=1:order
        kron_mul_elem_t!(p-1,order-p,ma,c,dvec,w)
    end
end

"""
    function kron_at_kron_b_mul_c!(a::AbstractMatrix, order::Int64, b::AbstractMatrix, c::AbstractVector, w::AbstractVector)
computes (a^T ⊗ a^T ⊗ ... ⊗ a^T ⊗ b)c
""" 
function kron_at_kron_b_mul_c!(a::AbstractMatrix, order::Int64, b::AbstractMatrix, c::AbstractVector, w::AbstractVector)
    m = size(b,2)
    n = size(a,2)
    w1 = convert(Matrix{Float64},reshape(w,m,n^order))
    c1 = convert(Matrix{Float64},reshape(c,m,n^order))
    A_mul_B!(w1,b,c1)
    copy!(c1,w1)
    for q = 0:order-1
        kron_mul_elem_t!(order-q-1,q,m,a,c,w)
    end
end

convert(::Type{Array{Float64, 2}}, x::Base.ReshapedArray{Float64,2,SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true},Tuple{}}) = unsafe_wrap(Array,pointer(x.parent.parent,x.parent.indexes[1][1]),x.dims)


"""
    kron_mul_elem(p::Int64, q::Int64, m::Int64, a::AbstractMatrix, b::AbstractVector, w::AbstractVector)

Performs (I_n^p \otimes a \otimes I_{m*n^q}) b, where n = size(a,2) and m, an arbitrary constant
"""
function kron_mul_elem!(p::Int64, q::Int64, m::Int64, a::AbstractMatrix, b::AbstractVector, w::AbstractVector)
    n = size(a,2)
    length(b) == m*n^(p+q+1) || throw(DimensionMismatch("The dimension of the vector, $(length(b)) doesn't correspond to order, ($p, $q)  and the dimension of the matrix, $(size(a))"))

    @inbounds begin
        if m == 1 && p + q == 0
            # a*b
            A_mul_B!(w,a,b)
            copy!(b,w)
        elseif m == 1 && q == 0
            #  (I_n^p ⊗ a)*b = vec(a*[b_1 b_2 ... b_p])
            np = n^p
            b = convert(Array{Float64,2},reshape(b,n,np))
            w = convert(Array{Float64,2},reshape(w,n,np))
            A_mul_B!(w,a,b)
            copy!(b,w)
        elseif p == 0
            # (a ⊗ I_{m*n^q})*b = (b'*(a' ⊗ I_{m*n^q}))' = vec(reshape(b,m*n^q,n)*a')
            mnq = m*n^q
            b = convert(Array{Float64,2},reshape(b,mnq,n))
            w = convert(Array{Float64,2},reshape(w,mnq,n))
            A_mul_Bt!(w,b,a)
            copy!(b,w)
        else
            # (I_{n^p} ⊗ a ⊗ I_{m*n^q})*b = vec([(a ⊗ I_{m*n^q})*b_1 (a ⊗ I_{m*n^q})*b_2 ... (a ⊗ I_{m*n^q})*b_{n^p}])
            mnq = m*n^q
            mnq1 = mnq*n
            qrange = 1:mnq1
            b_orig = copy(b)
            for i=1:n^p
                bi = convert(Array{Float64,2},reshape(view(b,qrange),mnq,n))
                wi = convert(Array{Float64,2},reshape(view(w,qrange),mnq,n))
                # (a ⊗ I_{m*n^q})*b = (b'*(a' ⊗ I_{m*n^q}))' = vec(reshape(b',m*n^q,n)*a')
                A_mul_Bt!(wi,bi,a)
                copy!(bi,wi)
                qrange += mnq1
            end
        end
    end
end

"""
    kron_mul_elem_t!(p::Int64, q::Int64, m::Int64, a::AbstractMatrix, b::AbstractVector, w::AbstractVector)

Performs (I_n^p \otimes a' \otimes I_{m*n^q}) b, where n = size(a,2) and m, an arbitrary constant
"""
function kron_mul_elem_t!(p::Int64, q::Int64, m::Int64, a::AbstractMatrix, b::AbstractVector, w::AbstractVector)
    n = size(a,2)
    length(b) == m*n^(p+q+1) || throw(DimensionMismatch("The dimension of the vector, $(length(b)) doesn't correspond to order, ($p, $q)  and the dimension of the matrix, $(size(a))"))

    @inbounds begin
        if m == 1 && p + q == 0
            # a'*b
            At_mul_B!(w,a,b)
            copy!(b,w)
        elseif m == 1 && q == 0
            #  (I_n^p ⊗ a')*b = vec(a*[b_1 b_2 ... b_p])
            np = n^p
            b = convert(Array{Float64,2},reshape(b,n,np))
            w = convert(Array{Float64,2},reshape(w,n,np))
            At_mul_B!(w,a,b)
            copy!(b,w)
        elseif p == 0
            # (a' ⊗ I_{m*n^q})*b = (b'*(a ⊗ I_{m*n^q}))' = vec(reshape(b,m*n^q,n)*a)
            mnq = m*n^q
            b = convert(Array{Float64,2},reshape(b,mnq,n))
            w = convert(Array{Float64,2},reshape(w,mnq,n))
            A_mul_B!(w,b,a)
            copy!(b,w)
        else
            # (I_{n^p} ⊗ a' ⊗ I_{m*n^q})*b = vec([(a' ⊗ I_{m*n^q})*b_1 (a' ⊗ I_{m*n^q})*b_2 ... (a' ⊗ I_{m*n^q})*b_{n^p}])
            mnq = m*n^q
            mnq1 = mnq*n
            qrange = 1:mnq1
            b_orig = copy(b)
            for i=1:n^p
                bi = convert(Array{Float64,2},reshape(view(b,qrange),mnq,n))
                wi = convert(Array{Float64,2},reshape(view(w,qrange),mnq,n))
                # (a' ⊗ I_{m*n^q})*b = (b'*(a ⊗ I_{m*n^q}))' = vec(reshape(b',m*n^q,n)*a)
                A_mul_B!(wi,bi,a)
                copy!(bi,wi)
                qrange += mnq1
            end
        end
    end
end

