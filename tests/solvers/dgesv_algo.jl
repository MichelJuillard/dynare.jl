include("dyn_blas_lapack_utils.jl")

type DgesvWS
    ipiv::Array{BlasInt,1}
    function DgesvWS(n)
        ipiv = Array(BlasInt,n)
        new(ipiv)
    end
end

function dgesv_core!(ws::DgesvWS,A::StridedMatrix{Float64},B::StridedMatrix{Float64})
    n = Ref{BlasInt}(size(A,1))
    m = Ref{BlasInt}(size(B,2))
    ldA = Ref{BlasInt}(max(1,stride(A,2)))
    ldB = Ref{BlasInt}(max(1,stride(B,2)))
    info = Ref{BlasInt}(0)
    ccall((@blasfunc(dgesv_), liblapack), Void,
          (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{BlasInt},Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt}),
          n, m, A, ldA, ws.ipiv, B, ldB, info)
    chklapackerror(info[])
end
