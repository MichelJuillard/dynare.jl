module QrAlgo

import LinearAlgebra: BlasInt
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra.BLAS: libblas
import LinearAlgebra.LAPACK: liblapack, chklapackerror

export QrWS, dgeqrf_core!, dormrqf_core!

struct QrWS
    tau::Vector{Float64}
    work::Vector{Float64}
    lwork::Ref{BlasInt}
    info::Ref{BlasInt}

    function QrWS(A::StridedMatrix{Float64})
        nn,mm = size(A)
        m = Ref{BlasInt}(mm)
        n = Ref{BlasInt}(nn)
        RldA = Ref{BlasInt}(max(1,stride(A,2)))
        tau = Vector{Float64}(undef, min(nn,mm))
        work = Vector{Float64}(undef, 1)
        lwork = Ref{BlasInt}(-1)
        info = Ref{BlasInt}(0)
        ccall((@blasfunc(dgeqrf_), liblapack), Nothing,
              (Ref{BlasInt},Ref{BlasInt},Ptr{Float64},Ref{BlasInt},
               Ptr{Float64},Ptr{Float64},Ref{BlasInt},Ref{BlasInt}),
              m,n,A,RldA,tau,work,lwork,info)
        chklapackerror(info[])
        lwork = Ref{BlasInt}(real(work[1]))
        work = Array{Float64}(undef, lwork[])
        new(tau,work,lwork,info)
    end
end

function dgeqrf_core!(ws::QrWS,A::StridedMatrix{Float64})
    mm,nn = size(A)
    m = Ref{BlasInt}(mm)
    n = Ref{BlasInt}(nn)
    RldA = Ref{BlasInt}(max(1,stride(A,2)))
    ccall((@blasfunc(dgeqrf_), liblapack), Nothing,
          (Ref{BlasInt},Ref{BlasInt},Ptr{Float64},Ref{BlasInt},
           Ptr{Float64},Ptr{Float64},Ref{BlasInt},Ref{BlasInt}),
          m,n,A,RldA,ws.tau,ws.work,ws.lwork,ws.info)
    chklapackerror(ws.info[])
end

function dormrqf_core!(ws::QrWS,side::Ref{UInt8},trans::Ref{UInt8},A::StridedMatrix{Float64},
                      C::StridedMatrix{Float64})
    mm,nn = size(C)
    m = Ref{BlasInt}(mm)
    n = Ref{BlasInt}(nn)
    k = Ref{BlasInt}(length(ws.tau))
    RldA = Ref{BlasInt}(max(1,stride(A,2)))
    RldC = Ref{BlasInt}(max(1,stride(C,2)))
    ccall((@blasfunc(dormqr_), liblapack), Nothing,
          (Ref{UInt8},Ref{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},Ptr{Float64},Ref{BlasInt},
           Ptr{Float64},Ptr{Float64},Ref{BlasInt},Ptr{Float64},Ref{BlasInt},Ref{BlasInt}),
          side,trans,m,n,k,A,RldA,ws.tau,C,RldC,ws.work,ws.lwork,ws.info)
    chklapackerror(ws.info[])
end
    
end 
                      

