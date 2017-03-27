module dgges_algo

# general Schur decomposition with reordering
# adataped from ./base/linalg/lapack.jl

import dyn_blas_lapack_utils: @blasfunc, BlasInt, chkstride1, checksquare, chklapackerror, liblapack
export DgeesWS, dgees!, DggesWS, dgges_core!

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
        Acopy = copy(A)
        ccall((@blasfunc(dgees_), liblapack), Void,
              (Ref{UInt8}, Ref{UInt8}, Ptr{Void},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
               Ptr{BlasInt}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ref{BlasInt},
               Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
               Ref{BlasInt}),
              jobvs, Rsort, mycompare_c,
              n, Acopy, RldA, 
              sdim, wr, wi,
              vs, ldvs,
              work, lwork, bwork,
              info)
        chklapackerror(info[])
        lwork = Ref{BlasInt}(real(work[1]))
        work = Array{Float64}(lwork[])
        eigen_values = Array{Complex{Float64}}(n[])
        new(jobvs, sdim, wr, wi, ldvs, vs, work, lwork, bwork, eigen_values, info)
    end

end


function DgeesWS(jobvs::Ref{UInt8}, A::StridedMatrix{Float64})
    chkstride1(A)
    checksquare(A)
    sdim = Ref{BlasInt}(0)
    n = size(A,1)
    wr = Array(Float64, n)
    wi = Array(Float64, n)
    ldvs = Ref{BlasInt}(jobvs[] == UInt32('V') ? n : 1)
    vs = Array(Float64, ldvs[], n)
    work = Array(Float64,1)
    lwork = Ref{BlasInt}(-1)
    bwork = Array(Int64,n)
    eigen_values = Array(Complex{Float64}, n)
    info = Ref{BlasInt}(0)
    DgeesWS(jobvs, A, sdim, wr, wi, ldvs, vs , work, lwork, bwork, eigen_values, info)

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
    ws.eigen_values = complex.(ws.wr,ws.wi)
    chklapackerror(ws.info[])
end

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
        Acopy = copy(A)
        ccall((@blasfunc(dgges_), liblapack), Void,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Void},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
               Ptr{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
               Ref{BlasInt}),
              jobvsl, jobvsr, Rsort, mycompare_g_c,
              n, Acopy, RldA, B,
              RldB, sdim, alphar, alphai,
              beta, vsl, ldvsl, vsr,
              ldvsr, work, lwork, bwork,
              info)
        chklapackerror(info[])
        lwork = Ref{BlasInt}(real(work[1]))
        work = Array{Float64}(lwork[])
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
    ldvsl = Ref{BlasInt}(jobvsl[] == UInt32('V') ? n : 1)
    vsl = Array(Float64, ldvsl[], n)
    ldvsr = Ref{BlasInt}(jobvsr[] == UInt32('V') ? n : 1)
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
          ws.jobvsl, ws.jobvsr, sort, mycompare_g_c,
          n, A, RldA, B,
          RldB, ws.sdim, ws.alphar, ws.alphai,
          ws.beta, ws.vsl, ws.ldvsl, ws.vsr,
          ws.ldvsr, ws.work, ws.lwork, ws.bwork,
          ws.info)
    ws.eigen_values = complex.(ws.alphar,ws.alphai)./ws.beta
    chklapackerror(ws.info[])
end


end
