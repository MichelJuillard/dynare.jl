include("dyn_blas_lapack_utils.jl")

type DgesvxWS
    AF::Array{Float64,2}
    ipiv::Array{BlasInt,1}
    R::Array{Float64,1}
    C::Array{Float64,1}
    ferr::Array{Float64,1}
    berr::Array{Float64,1}
    work::Array{Float64,1}
    iwork::Array{BlasInt,1}
    function DgesvxWS(n,m)
        AF = Array(Float64,n,n)
        ipiv = Array(BlasInt,n)
        R = Array(Float64,n)
        C = Array(Float64,n)
        ferr = Array(Float64,m)
        berr = Array(Float64,m)
        work = Array(Float64,4*n)
        iwork = Array(BlasInt,n)                     
        new(AF,ipiv,R,C,ferr,berr,work,iwork)
    end
end

function dgesvx_core!(ws::DgesvxWS,fact::Ref{UInt8},trans::Ref{UInt8},A::StridedMatrix{Float64},B::StridedMatrix{Float64},X::StridedMatrix{Float64})
    n = Ref{BlasInt}(size(A,1))
    m = Ref{BlasInt}(size(B,2))
    ldA = Ref{BlasInt}(max(1,stride(A,2)))
    ldAF = Ref{BlasInt}(max(1,stride(ws.AF,2)))
    ldB = Ref{BlasInt}(max(1,stride(B,2)))
    ldX = Ref{BlasInt}(max(1,stride(X,2)))
    rcond = Ref{Float64}(0)
    info = Ref{BlasInt}(0)
    equed = Ref{UInt8}('N')
    ccall((@blasfunc(dgesvx_), liblapack), Void,
          (Ref{UInt8},Ref{UInt8},Ref{BlasInt}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{BlasInt}, Ref{UInt8}, Ptr{Float64}, Ptr{Float64},Ptr{Float64},
           Ref{BlasInt}, Ptr{Float64},Ref{BlasInt},Ref{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},
           Ptr{BlasInt},Ref{BlasInt}),
          fact, trans, n, m, A, ldA, ws.AF, ldAF, ws.ipiv, equed, ws.R, ws.C,
          B, ldB, X, ldX, rcond, ws.ferr, ws.berr, ws.work, ws.iwork, info)
    if info[] <= n[]
        chklapackerror(info[])
    end
end
