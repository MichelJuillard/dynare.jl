# general Schur decomposition with reordering
# adataped from ./base/linalg/lapack.jl

module DGGES

import Base: USE_BLAS64
export dgges_ws, dgges_core

const liblapack = Base.liblapack_name

# Constants
    I

typealias BlasFloat Union{Float64,Float32,Complex128,Complex64}
typealias BlasReal Union{Float64,Float32}
typealias BlasComplex Union{Complex128,Complex64}

typealias BlasInt Int64

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

type DggesWS
    jobvsl::Ref{UInt8}
    jobvsr::Ref{UInt8}
    sdim::Ref{BlasInt}
    alphar::Array{Float64,1}
    alphai::Array{Float64,1}
    beta::Array{Float64,1}
    ldvsl::Ref{BlasInt}
    vsl::Array{Float64,2}
    ldvsr::Ref{BlasInt}
    vsr::Array{Float64,2}
    work::Array{Float64,1}
    lwork::Ref{BlasInt}
    bwork::Array{Int64,1}
    eigen_values::Array{Complex{Float64},1}
    info::Ref{BlasInt}

    function DggesWS(jobvsl::Ref{UInt8}, jobvsr::Ref{UInt8}, A::StridedMatrix{Float64}, B::StridedMatrix{Float64},sdim::Ref{BlasInt},
                      alphar::Array{Float64,1},alphai::Array{Float64,1},beta::Array{Float64,1},ldvsl::Ref{BlasInt},vsl::Array{Float64,2},
                      ldvsr::Ref{BlasInt},vsr::Array{Float64,2},work::Array{Float64,1},lwork::Ref{BlasInt},bwork::Array{Int64,1},
                      eigen_values::Array{Complex{Float64},1},info::Ref{BlasInt})
        n = Ref{BlasInt}(size(A,1))
        RldA = Ref{BlasInt}(max(1,stride(A,2)))
        RldB = Ref{BlasInt}(max(1,stride(B,2)))
        Rsort = Ref{UInt8}('N')
        ccall((@blasfunc(dgges_), liblapack), Void,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Void},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
               Ptr{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
               Ref{BlasInt}),
              jobvsl, jobvsr, Rsort, mycompare_c,
              n, A, RldA, B,
              RldB, sdim, alphar, alphai,
              beta, vsl, ldvsl, vsr,
              ldvsr, work, lwork, bwork,
              info)
        chklapackerror(info[])
        lwork = Ref{BlasInt}(real(work[1]))
        work = Array{Float64}(lwork[])
#        eigen_values = Array{Complex{Float64}}(n[])
        new(jobvsl,jobvsr,sdim,alphar,alphai,beta,ldvsl,vsl,ldvsr,vsr,work,lwork,bwork,eigen_values,info)
    end

end


function DggesWS(jobvsl::Ref{UInt8}, jobvsr::Ref{UInt8}, A::StridedMatrix{Float64}, B::StridedMatrix{Float64})
    chkstride1(A, B)
    n, m = checksquare(A, B)
    if n != m
        throw(DimensionMismatch("Dimensions of A, ($n,$n), and B, ($m,$m), must match"))
    end
    sdim = Ref{BlasInt}(0)
    alphar = Array(Float64, n)
    alphai = Array(Float64, n)
    beta = Array(Float64, n)
    ldvsl = Ref{BlasInt}(jobvsl[] == 'V' ? n : 1)
    vsl = Array(Float64, ldvsl[], n)
    ldvsr = Ref{BlasInt}(jobvsr[] == 'V' ? n : 1)
    vsr = Array(Float64, ldvsr[], n)
    work = Array(Float64,1)
    lwork = Ref{BlasInt}(-1)
    bwork = Array(Int64,n)
    eigen_values = Array(Complex{Float64}, n)
    info = Ref{BlasInt}(0)
    DggesWS(jobvsl, jobvsr, A, B,sdim,alphar,alphai,beta,ldvsl,vsl,ldvsr,vsr,work,lwork,bwork,eigen_values,info)

end



    
function dgges_core!(ws::DggesWS,A::StridedMatrix{Float64}, B::StridedMatrix{Float64})
    n = Ref{BlasInt}(size(A,1))
    RldA = Ref{BlasInt}(max(1,stride(A,2)))
    RldB = Ref{BlasInt}(max(1,stride(B,2)))
    sort = Ref{UInt8}('S')
    ccall((@blasfunc(dgges_), liblapack), Void,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Void},
           Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
           Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
           Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
           Ref{BlasInt}),
          ws.jobvsl, ws.jobvsr, sort, mycompare_c,
          n, A, RldA, B,
          RldB, ws.sdim, ws.alphar, ws.alphai,
          ws.beta, ws.vsl, ws.ldvsl, ws.vsr,
          ws.ldvsr, ws.work, ws.lwork, ws.bwork,
          ws.info)
    ws.eigen_values = complex(ws.alphar,ws.alphai)./ws.beta
    chklapackerror(ws.info[])
end

end

#using DGGES

#ws = DGGES.dgges_ws(Ref{UInt8}('N'),Ref{UInt8}('V'),eye(2),eye(2))
#DGGES.dgges_core!(ws,eye(2),eye(2))
#print(ws.eigen_values)
