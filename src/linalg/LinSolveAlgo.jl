module LinSolveAlgo

#import Base.LinAlg.BlasInt
import LinearAlgebra.BLAS.@blasfunc
#import Base.LinAlg.BLAS.libblas
import LinearAlgebra.LAPACK: liblapack, chklapackerror
using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK

export LinSolveWS, linsolve_core!, linsolve_core_no_lu!, lu!

struct LinSolveWS
    lu::Matrix{Float64}
    ipiv::Vector{BLAS.BlasInt}

    function LinSolveWS(n)
        lu = Matrix{Float64}(undef, n,n)
        ipiv = Vector{BLAS.BlasInt}(undef, n)
        new(lu,ipiv)
    end
end

function linsolve_core_no_lu!(ws::LinSolveWS,trans::Ref{UInt8},a::StridedMatrix{Float64},b::StridedVecOrMat{Float64})
    mm,nn = size(a)
    m = Ref{BLAS.BlasInt}(mm)
    n = Ref{BLAS.BlasInt}(nn)
    nhrs = Ref{BLAS.BlasInt}(size(b,2))
    lda = Ref{BLAS.BlasInt}(max(1,stride(a,2)))
    ldb = Ref{BLAS.BlasInt}(max(1,stride(b,2)))
    info = Ref{BLAS.BlasInt}(0)

    ccall((@blasfunc(dgetrs_), liblapack), Nothing,
          (Ref{UInt8},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},
           Ptr{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt}),
          trans,n,nhrs,ws.lu,lda,ws.ipiv,b,ldb,info)
    if info[] != 0
        println("dgetrs ",info[])
        chklapackerror(info[])
    end
end

function linsolve_core!(ws::LinSolveWS,trans::Ref{UInt8},a::StridedMatrix{Float64},b::StridedVecOrMat{Float64})
    mm,nn = size(a)
    m = Ref{BLAS.BlasInt}(mm)
    n = Ref{BLAS.BlasInt}(nn)
    nhrs = Ref{BLAS.BlasInt}(size(b,2))
    lda = Ref{BLAS.BlasInt}(max(1,stride(a,2)))
    ldb = Ref{BLAS.BlasInt}(max(1,stride(b,2)))
    info = Ref{BLAS.BlasInt}(0)

    lu!(ws.lu,a,ws.ipiv)
    ccall((@blasfunc(dgetrs_), liblapack), Nothing,
          (Ref{UInt8},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},
           Ptr{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt}),
          trans,n,nhrs,ws.lu,lda,ws.ipiv,b,ldb,info)
    if info[] != 0
        println("dgetrs ",info[])
        chklapackerror(info[])
    end
end

function linsolve_core!(ws::LinSolveWS,trans::Ref{UInt8},a::StridedMatrix{Float64},b::StridedVecOrMat{Float64},c::StridedVecOrMat{Float64})
    mm,nn = size(a)
    m = Ref{BLAS.BlasInt}(mm)
    n = Ref{BLAS.BlasInt}(nn)
    lda = Ref{BLAS.BlasInt}(max(1,stride(a,2)))
    ldb = Ref{BLAS.BlasInt}(max(1,stride(b,2)))
    ldc = Ref{BLAS.BlasInt}(max(1,stride(c,2)))
    info = Ref{BLAS.BlasInt}(0)

    lu!(ws.lu,a,ws.ipiv)
    nhrs = Ref{BLAS.BlasInt}(size(b,2))
    ccall((@blasfunc(dgetrs_), liblapack), Nothing,
          (Ref{UInt8},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},
           Ptr{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt}),
          trans,n,nhrs,ws.lu,lda,ws.ipiv,b,ldb,info)
    if info[] != 0
        println("dgetrs ",info[])
        chklapackerror(info[])
    end
    nhrs = Ref{BLAS.BlasInt}(size(c,2))
    ccall((@blasfunc(dgetrs_), liblapack), Nothing,
          (Ref{UInt8},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},
           Ptr{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt}),
          trans,n,nhrs,ws.lu,lda,ws.ipiv,c,ldc,info)
    if info[] != 0
        println("dgetrs ",info[])
        chklapackerror(info[])
    end
end

function linsolve_core!(ws::LinSolveWS,trans::Ref{UInt8},a::StridedMatrix{Float64},b::StridedVecOrMat{Float64},c::StridedVecOrMat{Float64},d::StridedVecOrMat{Float64})
    mm,nn = size(a)
    m = Ref{BLAS.BlasInt}(mm)
    n = Ref{BLAS.BlasInt}(nn)
    nhrs = Ref{BLAS.BlasInt}(size(b,2))
    lda = Ref{BLAS.BlasInt}(max(1,stride(a,2)))
    ldb = Ref{BLAS.BlasInt}(max(1,stride(b,2)))
    ldc = Ref{BLAS.BlasInt}(max(1,stride(c,2)))
    ldd = Ref{BLAS.BlasInt}(max(1,stride(d,2)))
    info = Ref{BLAS.BlasInt}(0)

    lu!(ws.lu,a,ws.ipiv)
    ccall((@blasfunc(dgetrs_), liblapack), Nothing,
          (Ref{UInt8},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},
           Ptr{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt}),
          trans,n,nhrs,ws.lu,lda,ws.ipiv,b,ldb,info)
    if info[] != 0
        println("dgetrs ",info[])
        chklapackerror(info[])
    end
    ccall((@blasfunc(dgetrs_), liblapack), Nothing,
          (Ref{UInt8},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},
           Ptr{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt}),
          trans,n,nhrs,ws.lu,lda,ws.ipiv,c,ldc,info)
    if info[] != 0
        println("dgetrs ",info[])
        chklapackerror(info[])
    end
    ccall((@blasfunc(dgetrs_), liblapack), Nothing,
          (Ref{UInt8},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},
           Ptr{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},Ref{BLAS.BlasInt}),
          trans,n,nhrs,ws.lu,lda,ws.ipiv,d,ldd,info)
    if info[] != 0
        println("dgetrs ",info[])
        chklapackerror(info[])
    end
end

function lu!(lu,a,ipiv)
    copyto!(lu,a)
    mm,nn = size(a)
    m = Ref{BLAS.BlasInt}(mm)
    n = Ref{BLAS.BlasInt}(nn)
    lda = Ref{BLAS.BlasInt}(max(1,stride(a,2)))
    info = Ref{BLAS.BlasInt}(0)
    ccall((@blasfunc(dgetrf_), liblapack), Nothing,
          (Ref{BLAS.BlasInt},Ref{BLAS.BlasInt},Ptr{Float64},Ref{BLAS.BlasInt},
           Ptr{BLAS.BlasInt},Ref{BLAS.BlasInt}),
          m,n,lu,lda,ipiv,info)
    if info[] != 0
        println("dgetrf ",info[])
        chklapackerror(info[])
    end
end

end
