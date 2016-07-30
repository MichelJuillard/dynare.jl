include("dyn_blas_lapack_utils.jl")

type LinSolveWS
    ipiv::Array{BlasInt}

    function LinSolveWS(n)
        ipiv = Array(BlasInt,n)
        new(ipiv)
    end
end

function linsolve_core!(ws::LinSolveWS,trans::Ref{UInt8},a::StridedMatrix{Float64},b::StridedMatrix{Float64})
    mm,nn = size(a)
    m = Ref{BlasInt}(mm)
    n = Ref{BlasInt}(nn)
    nhrs = Ref{BlasInt}(size(b,2))
    lda = Ref{BlasInt}(max(1,stride(a,2)))
    ldb = Ref{BlasInt}(max(1,stride(b,2)))
    info = Ref{BlasInt}(0)
    ccall((@blasfunc(dgetrf_), liblapack), Void,
          (Ref{BlasInt},Ref{BlasInt},Ptr{Float64},Ref{BlasInt},
           Ptr{BlasInt},Ref{BlasInt}),
          m,n,a,lda,ws.ipiv,info)
    if info[] != 0
        println("dgetrf ",info[])
        chklapackerror(info[])
    end

    ccall((@blasfunc(dgetrs_), liblapack), Void,
          (Ref{UInt8},Ref{BlasInt},Ref{BlasInt},Ptr{Float64},Ref{BlasInt},
           Ptr{BlasInt},Ptr{Float64},Ref{BlasInt},Ref{BlasInt}),
          trans,n,nhrs,a,lda,ws.ipiv,b,ldb,info)
    if info[] != 0
        println("dgetrs ",info[])
        chklapackerror(info[])
    end
end
