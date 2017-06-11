module DGGES

# general Schur decomposition with reordering
# adataped from ./base/linalg/lapack.jl

include("exceptions.jl")

import Base: USE_BLAS64
export DgeesWS, dgees!, DggesWS, dgges!

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

function mycompare{T}(wr_::Ptr{T}, wi_::Ptr{T})
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return convert(Cint, ((wr * wr + wi * wi) < criterium) ? 1 : 0)
end

function mycompare{T}(alphar_::Ptr{T}, alphai_::Ptr{T}, beta_::Ptr{T})
    alphar = unsafe_load(alphar_)
    alphai = unsafe_load(alphai_)
    beta = unsafe_load(beta_)
    return convert(Cint, ((alphar * alphar + alphai * alphai) < criterium * beta * beta) ? 1 : 0)
end

const mycompare_c = cfunction(mycompare, Cint, (Ptr{Cdouble}, Ptr{Cdouble}))
const mycompare_g_c = cfunction(mycompare, Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))

type DgeesWS
    jobvs::Ref{UInt8}
    sdim::Ref{BlasInt}
    wr::Array{Float64,1}
    wi::Array{Float64,1}
    ldvs::Ref{BlasInt}
    vs::Array{Float64,2}
    work::Array{Float64,1}
    lwork::Ref{BlasInt}
    bwork::Array{Int64,1}
    eigen_values::Array{Complex{Float64},1}
    info::Ref{BlasInt}

    function DgeesWS(jobvs::Ref{UInt8}, A::StridedMatrix{Float64}, sdim::Ref{BlasInt},
                      wr::Array{Float64,1}, wi::Array{Float64,1}, ldvs::Ref{BlasInt}, vs::Array{Float64,2},
                      work::Array{Float64,1}, lwork::Ref{BlasInt}, bwork::Array{Int64,1},
                      eigen_values::Array{Complex{Float64},1}, info::Ref{BlasInt})
        n = Ref{BlasInt}(size(A,1))
        RldA = Ref{BlasInt}(max(1,stride(A,2)))
        Rsort = Ref{UInt8}('N')
        ccall((@blasfunc(dgees_), liblapack), Void,
              (Ref{UInt8}, Ref{UInt8}, Ptr{Void},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
               Ptr{BlasInt}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ref{BlasInt},
               Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
               Ref{BlasInt}),
              jobvs, Rsort, mycompare_c,
              n, A, RldA, 
              sdim, wr, wi,
              vs, ldvs,
              work, lwork, bwork,
              info)
        chklapackerror(info[])
        lwork = Ref{BlasInt}(real(work[1]))
        work = Array{Float64}(lwork[])
#        eigen_values = Array{Complex{Float64}}(n[])
        new(jobvs, sdim, wr, wi, ldvs, vs, work, lwork, bwork, eigen_values, info)
    end

end


function DgeesWS(A::StridedMatrix{Float64})
    chkstride1(A)
    n, = checksquare(A)
    jobvs = Ref{UInt8}('V')
    sdim = Ref{BlasInt}(0)
    wr = Array(Float64, n)
    wi = Array(Float64, n)
    ldvs = Ref{BlasInt}(jobvs[] == UInt32('V') ? n : 1)
    vs = Array(Float64, ldvs[], n)
    work = Array(Float64,1)
    lwork = Ref{BlasInt}(-1)
    bwork = Array(Int64,n)
    eigen_values = Array(Complex{Float64}, n)
    info = Ref{BlasInt}(0)
    DgeesWS(jobvs, A, sdim, wr, wi, ldvs, vs, work, lwork, bwork, eigen_values, info)

end
    
function dgees!(ws::DgeesWS,A::StridedMatrix{Float64})
    n = Ref{BlasInt}(size(A,1))
    RldA = Ref{BlasInt}(max(1,stride(A,2)))
    sort = Ref{UInt8}('S')
    ccall((@blasfunc(dgees_), liblapack), Void,
          (Ref{UInt8}, Ref{UInt8}, Ptr{Void},
           Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
           Ref{BlasInt}),
          ws.jobvs, sort, mycompare_c,
          n, A, RldA,
          ws.sdim, ws.wr, ws.wi,
          ws.vs, ws.ldvs,
          ws.work, ws.lwork, ws.bwork,
          ws.info)
    ws.eigen_values = complex.(ws.wr, ws.wi)
    chklapackerror(ws.info[])
end

type DggesWS
    alphar::Array{Float64,1}
    alphai::Array{Float64,1}
    beta::Array{Float64,1}
    lwork::BlasInt
    work::Array{Float64,1}
    bwork::Array{Int64,1}
    sdim::BlasInt

    function DggesWS(A,B)
        chkstride1(A, B)
        n, m = checksquare(A, B)
        if n != m
            throw(DimensionMismatch("Dimensions of A, ($n,$n), and B, ($m,$m), must match"))
        end
        n = BlasInt(size(A,1))
        alphar = Array(Float64,n)
        alphai = Array(Float64,n)
        beta = Array(Float64,n)
        bwork = Array(Int64,n)
        jobvsl = 'N'
        jobvsr = 'N'
        ldvsl = BlasInt(1)
        ldvsr = BlasInt(1)
        sdim = BlasInt(0)
        sort = 'N'
        lwork = BlasInt(-1)
        work = Array(Float64,1)
        sdim = BlasInt(0)
        info = BlasInt(0)
        ccall((@blasfunc(dgges_), liblapack), Void,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Void},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
               Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
               Ref{BlasInt}),
              jobvsl, jobvsr, sort, mycompare_c,
              n, A, max(1,stride(A, 2)), B,
              max(1,stride(B, 2)), sdim, alphar, alphai,
              beta, C_NULL, ldvsl, C_NULL,
              ldvsr, work, lwork, bwork,
              info)
        chklapackerror(info)
        lwork = BlasInt(real(work[1]))
        work = Array{Float64}(lwork)
        new(alphar,alphai,beta,lwork,work,bwork,sdim)
    end
end

function dgges!(jobvsl::Char, jobvsr::Char, A::StridedMatrix{Float64}, B::StridedMatrix{Float64},
                vsl::Array{Float64,2}, vsr::Array{Float64,2}, eigval::Array{Complex64,1},
                ws::DggesWS)
    n = size(A,1)
    ldvsl = jobvsl == 'V' ? n : 1
    ldvsr = jobvsr == 'V' ? n : 1
    sort = 'S'
    sdim = Ref{BlasInt}(0)
    info = Ref{BlasInt}(0)
    ccall((@blasfunc(dgges_), liblapack), Void,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Void},
           Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
           Ref{BlasInt}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
           Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
           Ref{Int64}),
          jobvsl, jobvsr, sort, mycompare_c,
          n, A, max(1,stride(A, 2)), B,
          max(1,stride(B, 2)), sdim, ws.alphar, ws.alphai,
          ws.beta, vsl, ldvsl, vsr,
          ldvsr, ws.work, ws.lwork, ws.bwork,
          info)
    ws.sdim = sdim[]
    if info[]
        throw(DggesException(info[]))
    end
    for i in 1:n
        eigval[i] = complex(ws.alphar[i],ws.alphai[i])/ws.beta[i]
    end
end

end

