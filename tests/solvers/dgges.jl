# general Schur decomposition with reordering
# adataped from ./base/linalg/lapack.jl

module DGGES

import Base: USE_BLAS64
export dgges!

const liblapack = Base.liblapack_name

# Constants
    I

typealias BlasFloat Union{Float64,Float32,Complex128,Complex64}
typealias BlasReal Union{Float64,Float32}
typealias BlasComplex Union{Complex128,Complex64}

if USE_BLAS64
    typealias BlasInt Int64
else
    typealias BlasInt Int32
end

# utility routines
function vendor()
    try
        cglobal((:openblas_set_num_threads, Base.libblas_name), Void)
        return :openblas
    end
    try
        cglobal((:openblas_set_num_threads64_, Base.libblas_name), Void)
        return :openblas64
    end
    try
        cglobal((:MKL_Set_Num_Threads, Base.libblas_name), Void)
        return :mkl
    end
    return :unknown
end

if vendor() == :openblas64
    macro blasfunc(x)
        return Expr(:quote, Symbol(x, "64_"))
    end
    openblas_get_config() = strip(unsafe_string(ccall((:openblas_get_config64_, Base.libblas_name), Ptr{UInt8}, () )))
else
    macro blasfunc(x)
        return Expr(:quote, x)
    end
    openblas_get_config() = strip(unsafe_string(ccall((:openblas_get_config, Base.libblas_name), Ptr{UInt8}, () )))
end

# Check that stride of matrix/vector is 1
# Writing like this to avoid splatting penalty when called with multiple arguments,
# see PR 16416
@inline chkstride1(A...) = _chkstride1(true, A...)
@noinline _chkstride1(ok::Bool) = ok || error("matrix does not have contiguous columns")
@inline _chkstride1(ok::Bool, A, B...) = _chkstride1(ok & (stride(A, 1) == 1), B...)

"""
    LinAlg.checksquare(A)

Check that a matrix is square, then return its common dimension. For multiple arguments, return a vector.
"""
function checksquare(A)
    m,n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square"))
    m
end

function checksquare(A...)
    sizes = Int[]
    for a in A
        size(a,1)==size(a,2) || throw(DimensionMismatch("matrix is not square: dimensions are $(size(a))"))
        push!(sizes, size(a,1))
    end
    return sizes
end

"Handle all nonzero info codes"
function chklapackerror(ret::BlasInt)
    if ret == 0
        return
    elseif ret < 0
        throw(ArgumentError("invalid argument #$(-ret) to LAPACK call"))
    else # ret > 0
        throw(LAPACKException(ret))
    end
end

const criterium = 1+1e-6

function mycompare{T}(alphar_::Ptr{T}, alphai_::Ptr{T}, beta_::Ptr{T})
    alphar = unsafe_load(alphar_)
    alphai = unsafe_load(alphai_)
    beta = unsafe_load(beta_)
    return convert(Cint, ((alphar * alphar + alphai * alphai) < criterium * beta * beta) ? 1 : 0)
end

const mycompare_c = cfunction(mycompare, Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))

function dgges!(jobvsl::Char, jobvsr::Char, A::StridedMatrix{Float64}, B::StridedMatrix{Float64})
    chkstride1(A, B)
    n, m = checksquare(A, B)
    if n != m
        throw(DimensionMismatch("Dimensions of A, ($n,$n), and B, ($m,$m), must match"))
    end
    sdim = Ref{BlasInt}()
    alphar = similar(A, Float64, n)
    alphai = similar(A, Float64, n)
    beta = similar(A, Float64, n)
    ldvsl = jobvsl == 'V' ? n : 1
    vsl = similar(A, Float64, ldvsl, n)
    ldvsr = jobvsr == 'V' ? n : 1
    vsr = similar(A, Float64, ldvsr, n)
    work = Array{Float64}(1)
    lwork = BlasInt(-1)
    bwork = Array{Int64}(n)
    sort = 'N'
    info = Ref{BlasInt}()
    for i = 1:2
        ccall((@blasfunc(dgges_), liblapack), Void,
              (Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{Void},
               Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64},
               Ptr{BlasInt}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ptr{BlasInt}, Ptr{Float64},
               Ptr{BlasInt}, Ptr{Float64}, Ptr{BlasInt}, Ptr{Int64},
               Ptr{BlasInt}),
              &jobvsl, &jobvsr, &sort, mycompare_c,
              &n, A, &max(1,stride(A, 2)), B,
              &max(1,stride(B, 2)), sdim, alphar, alphai,
              beta, vsl, &ldvsl, vsr,
              &ldvsr, work, &lwork, bwork,
              info)
        chklapackerror(info[])
        if i == 1
            sort = 'S'
            lwork = BlasInt(real(work[1]))
            work = Array{Float64}(lwork)
        end
    end
    A, B, complex(alphar, alphai)./beta, sdim[], vsl[1:(jobvsl == 'V' ? n : 0),:], vsr[1:(jobvsr == 'V' ? n : 0),:]
end

end
