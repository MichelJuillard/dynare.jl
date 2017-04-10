## QuasiUpperTriangular Matrix of real numbers

using Base.Test

import Base.size
import Base.similar
import Base.getindex
import Base.setindex!
import Base.copy

immutable QuasiUpperTriangular{T<:Real,S<:AbstractMatrix} <: AbstractMatrix{T}
    data::S
end
QuasiUpperTriangular(A::QuasiUpperTriangular) = A
function QuasiUpperTriangular(A::AbstractMatrix)
    Base.LinAlg.checksquare(A)
    return QuasiUpperTriangular{eltype(A), typeof(A)}(A)
end

size(A::QuasiUpperTriangular, d) = size(A.data, d)
size(A::QuasiUpperTriangular) = size(A.data)

convert{T,S}(::Type{QuasiUpperTriangular{T}}, A::QuasiUpperTriangular{T,S}) = A
function convert{Tnew,Told,S}(::Type{QuasiUpperTriangular{Tnew}}, A::QuasiUpperTriangular{Told,S})
    Anew = convert(AbstractMatrix{Tnew}, A.data)
    QuasiUpperTriangular(Anew)
end
convert{Tnew,Told,S}(::Type{AbstractMatrix{Tnew}}, A::QuasiUpperTriangular{Told,S}) = convert(QuasiUpperTriangular{Tnew}, A)
convert{T,S}(::Type{Matrix}, A::QuasiUpperTriangular{T,S}) = convert(Matrix{T}, A)

function similar{T,S,Tnew}(A::QuasiUpperTriangular{T,S}, ::Type{Tnew})
    B = similar(A.data, Tnew)
    return QuasiUpperTriangular(B)
end

copy(A::QuasiUpperTriangular) = QuasiUpperTriangular(copy(A.data))

broadcast(::typeof(big), A::QuasiUpperTriangular) = QuasiUpperTriangular(big.(A.data))

real{T<:Real}(A::QuasiUpperTriangular{T}) = A
broadcast(::typeof(abs), A::QuasiUpperTriangular) = QuasiUpperTriangular(abs.(A.data))

getindex{T,S}(A::QuasiUpperTriangular{T,S}, i::Integer, j::Integer) =
    i <= j + 1 ? A.data[i,j] : zero(A.data[j,i])

function setindex!(A::QuasiUpperTriangular, x, i::Integer, j::Integer)
    if i > j + 1
        x == 0 || throw(ArgumentError("cannot set index in the lower triangular part " *
            "($i, $j) of an UpperTriangular matrix to a nonzero value ($x)"))
    else
        A.data[i,j] = x
    end
    return A
end

## Generic quasi triangular multiplication
function A_mul_B!(A::QuasiUpperTriangular, B::AbstractVecOrMat)
    m, n = size(B, 1), size(B, 2)
    if m != size(A, 1)
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    for j = 1:n
        Bij2 = A.data[1,1]*B[1,j]
        for k = 2:m
            Bij2 += A.data[1,k]*B[k,j]
        end
        for i = 2:m
            Bij1 = A.data[i,i-1]*B[i-1,j]
            for k = i:m
                Bij1 += A.data[i,k]*B[k,j]
            end
            B[i-1,j] = Bij2
            Bij2 = Bij1
        end
        B[m,j] = Bij2
    end
    B
end

function At_mul_B!(A::QuasiUpperTriangular, B::AbstractVecOrMat)
    m, n = size(B, 1), size(B, 2)
    if m != size(A, 1)
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    for j = 1:n
        Bij2 = A.data[m,m].'B[m,j]
        for k = 1:m - 1
            Bij2 += A.data[k,m].'B[k,j]
        end
        for i = m-1:-1:1
            Bij1 = A.data[i+1,i].'B[i+1,j]
            for k = 1:i
                Bij1 += A.data[k,i].'B[k,j]
            end
            B[i+1,j] = Bij2
            Bij2 = Bij1
        end
        B[1,j] = Bij2
    end
    B
end

function A_mul_B!(A::AbstractMatrix, B::QuasiUpperTriangular)
    m, n = size(A)
    if size(B, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    for i = 1:m
        Aij2 = A[i,n]*B.data[n,n]
        for k = 1:n - 1
            Aij2 += A[i,k]*B.data[k,n]
        end
        for j = n-1:-1:1
            Aij1 = A[i,j+1]*B.data[j+1,j]
            for k = 1:j
                Aij1 += A[i,k]*B.data[k,j]
            end
            A[i,j+1] = Aij2
            Aij2 = Aij1
        end
        A[i,1] = Aij2
    end
    A
end

function A_mul_Bt!(A::AbstractMatrix, B::QuasiUpperTriangular)
    m, n = size(A)
    if size(B, 1) != n
        throw(DimensionMismatch("right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    for i = 1:m
        Aij2 = A[i,1]*B.data[1,1].'
        for k = 2:n
            Aij2 += A[i,k]*B.data[1,k].'
        end
        for j = 2:n
            Aij1 = A[i,j-1]*B.data[j,j-1].'
            for k = j:n
                Aij1 += A[i,k]*B.data[j,k].'
            end
            A[i,j-1] = Aij2
            Aij2 = Aij1
        end
        A[i,n] = Aij2
    end
    A
end

# solver by substitution
function A_ldiv_B!(a::QuasiUpperTriangular, b::AbstractMatrix)
    m, n = size(a)
    nb, p = size(b)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    j = n
    while j > 0
        if j == 1 || a.data[j,j-1] == 0
            a.data[j,j] == zero(a.data[j,j]) && throw(SingularException(j))
            for k = 1:p
                xj = b[j, k] = a.data[j,j] \ b[j, k]
                for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                    b[i, k] -= a.data[i,j] * xj
                end
            end
            j -= 1
        else
            a11, a21, a12, a22 = a.data[j-1:j,j-1:j]
            det = a11*a22 - a12*a21
            det == zero(a.data[j,j]) && throw(SingularException(j))
            m1 = -a21/det
            m2 = a11/a21
            for k = 1:p
                x2 = m1 * (b[j-1,k] - m2 * b[j,k])
                x1 = b[j-1,k] = a21 \ (b[j,k] - a22 * x2)
                b[j,k] = x2
                for i in j-2:-1:1
                    b[i,k] -= a.data[i,j-1] * x1 + a.data[i,j] * x2
                end
            end
            j -= 2
        end
    end
end

    
function A_rdiv_B!(a::AbstractMatrix, b::QuasiUpperTriangular)
    m, n = size(a)
    nb, p = size(b)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    for i = 1:m
        j = 1
        while  j <= n
            if j == n || b.data[j+1,j] == 0
                b.data[j,j] == zero(b.data[j,j]) && throw(SingularException(j))
                aij = a[i,j]
                for k in 1:j-1
                   aij -= a[i,k] * b.data[k,j]
                end
                a[i,j] = aij/b.data[j,j]
                j += 1
            else
                b11, b21, b12, b22 = b.data[j:j+1,j:j+1]
                det = b11*b22 - b12*b21
                det == zero(b.data[j,j]) && throw(SingularException(j))
                a1 = a[i,j]
                a2 = a[i,j+1]
                for k in 1:j-1
                    a1 -= a[i,k]*b.data[k,j]
                    a2 -= a[i,k]*b.data[k,j+1]
                end
                m1 = -b21/det
                m2 = b22/b21
                a[i,j] = m1 * (a2 - m2 * a1)
                a[i,j+1] = b21 \ (a1 - b11 * a[i,j])
                j += 2
            end
        end
    end
end

function A_rdiv_Bt!(a::AbstractMatrix, b::QuasiUpperTriangular)
    x=a/b.'
    m, n = size(a)
    nb, p = size(b)
    if nb != n
        throw(DimensionMismatch("right hand side b needs first dimension of size $n, has size $(size(b,1))"))
    end
    for i = 1:m
        j = n
        while  j > 0
            if j == 1 || b.data[j,j-1] == 0
                b.data[j,j] == zero(b.data[j,j]) && throw(SingularException(j))
                aij = a[i,j]
                for k = j + 1:n
                    aij -= a[i,k] * b.data[j,k]
                end
                a[i,j] = aij/b.data[j,j]
                j -= 1
            else
                b11, b21, b12, b22 = b.data[j-1:j,j-1:j]
                det = b11*b22 - b12*b21
                det == zero(b.data[j,j]) && throw(SingularException(j))
                a1 = a[i,j-1]
                a2 = a[i,j]
                for k in j + 1:n
                    a1 -= a[i,k]*b.data[j-1,k]
                    a2 -= a[i,k]*b.data[j,k]
                end
                m1 = -b21/det
                m2 = b11/b21
                a[i,j] = m1 * (a1 - m2 * a2)
                a[i,j-1] = b21 \ (a2 - b22 * a[i,j])
                j -= 2
            end
        end
    end
end
