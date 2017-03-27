import Base.A_mul_B!
import Base.A_mul_Bt!
import Base.At_mul_B!

include("quasi_upper_triangular.jl")

using dgges_algo
using linsolve_algo

type DiscreteLyapunovWS
    dgees_ws::DgeesWS
    linsolve_ws::LinSolveWS
    t:: QuasiUpperTriangular
    c::Array{Float64,2}
    w1:: QuasiUpperTriangular
    b1:: Array{Float64,2}
    w2::Array{Float64,2}
    b2::Array{Float64,1}
    b2n::Array{Float64,2}
    b::Array{Float64,2}
    temp::Array{Float64,2}
    function DiscreteLyapunovWS(a,m,n)
        dgees_ws = DgeesWS(Ref{UInt8}('V'),a)
        linsolve_ws = LinSolveWS(2*n)
        t = QuasiUpperTriangular(zeros(n,n))
        c = Array(Float64,m,n)
        w1 = QuasiUpperTriangular(zeros(n,n))
        b1 = Array(Float64,1,n)
        w2 = Array(Float64,2*n,2*n)
        b2 = Array(Float64,2*n)
        b2n = Array(Float64,2,n)
        b = Array(Float64,m,n)
        temp = Array(Float64,m,n)
        new(dgees_ws,linsolve_ws,t,c,w1,b1,w2,b2,b,temp)
    end
end

function discrete_lyapunov_solver!(ws::DiscreteLyapunovWS,a::Array{Float64,2},b::Array{Float64,2},x::Array{Float64,2})
    n = size(a,1)
    dgees!(ws.dgees_ws,a)
    ws.t = QuasiUpperTriangular(a)
    ws.b = copy(b)
    At_mul_B!(ws.b,ws.dgees_ws.vs,b)
    A_mul_B!(ws.c,ws.b,ws.dgees_ws.vs)

    i = n
    while i > 0
        if i == 1 || ws.t[i,i-1] == 0
            solve_one_row!(i,n,ws.t,ws.c,ws.w1,ws.b1)
            i -= 1
        else
            solve_two_rows!(ws.linsolve_ws,i,n,ws.t,ws.c,ws.w2,ws.b2,ws.b2n)
            i -= 2
        end
    end

    A_mul_B!(ws.b,ws.dgees_ws.vs,ws.c)
    A_mul_Bt!(x,ws.b,ws.dgees_ws.vs)
    x
end

function discrete_lyapunov_symmetrical_solver!(ws::DiscreteLyapunovWS,a::Array{Float64,2},b::Array{Float64,2},x::Array{Float64,2})
    n = size(a,1)
    dgees!(ws.dgees_ws,a)
    ws.t = QuasiUpperTriangular(a)
    ws.b = copy(b)
    At_mul_B!(ws.b,ws.dgees_ws.vs,b)
    A_mul_B!(ws.c,ws.b,ws.dgees_ws.vs)

    i = n
    while i > 0
        ti = QuasiUpperTriangular(view(ws.t,1:i,1:i))
        ci = view(ws.c,1:i,1:i)
        if i == 1 || ws.t[i,i-1] == 0
            w1i = QuasiUpperTriangular(view(ws.w1,1:i,1:i))
            b1i = view(ws.b1,:,1:i)
            solve_one_row!(i,i,ti,ci,w1i,b1i)
            for j=1:i-1
                ws.c[j,i] = ci[i,j]
                for k=1:i-1
                    ws.b[k,j] = ci[i,k]*ti[j,i]
                end
            end
            for j=1:i-1
                for m=1:i-1
                    ws.c[1,j] += ws.t[1,m]*ws.b[m,j]
                end
                for k=2:i-1
                    for m=k-1:i-1
                        ws.c[k,j] += ws.t[k,m]*ws.b[m,j]
                    end
                end
            end
            i -= 1
        else
            w2i = view(ws.w2,1:2*i,1:2*i)
            b2i = view(ws.b2,1:2*i)
            b2ni = view(ws.b2n,:,1:i)
            solve_two_rows!(ws.linsolve_ws,i,i,ti,ci,w2i,b2i,b2ni)
            for j=1:i-2
                ws.c[j,i-1] = ci[i-1,j]
                ws.c[j,i] = ci[i,j]
                for k=1:i-1
                    ws.b[k,j] = ci[i-1,k]*ti[j,i-1] + ci[i,k]*ti[j,i]
                end
            end
            for j=1:i-2
                for m=1:i-2
                    ws.c[1,j] += ws.t[1,m]*ws.b[m,j]
                end
                for k=2:i-2
                    for m=k-1:i-2
                        ws.c[k,j] += ws.t[k,m]*ws.b[m,j]
                    end
                end
            end
            i -= 2
        end
    end
    
    A_mul_B!(ws.b,ws.dgees_ws.vs,ws.c)
    A_mul_Bt!(x,ws.b,ws.dgees_ws.vs)
    x
end

function solve_one_row!(i::Int64,n::Int64,t::QuasiUpperTriangular,b::StridedMatrix,w::QuasiUpperTriangular,b1::StridedMatrix)
    tau = t[i,i]
    for j = 1:n-1
        for k = 1:j-1
            w[k,j] = - tau * t[k,j]
        end
        w[j,j] = 1.0 - tau * t[j,j]
        w[j+1,j] = - tau * t[j+1,j]
    end
    for k = 1:n-1
        w[k,n] = - tau * t[k,n]
    end
    w[n,n] = 1.0 - tau * t[n,n]
    bi = view(b,i:i,:)
    A_rdiv_Bt!(bi,w)
    b1 = copy(bi)
    A_mul_Bt!(b1,t)
    for k = 1:n
        for j = 1:i-1
            b[j,k] += t[j,i]*b1[1,k]
        end
    end
end

function solve_two_rows!(linsolve_ws::LinSolveWS,i::Int64,n::Int64,t::QuasiUpperTriangular,b::StridedMatrix,w::StridedMatrix,b2::StridedVector,b2n::StridedMatrix)
    tau11 = t[i-1,i-1]
    tau21 = t[i,i-1]
    tau12 = t[i-1,i]
    tau22 = t[i,i]
    fill!(w,0.0)
    w[1,1] = 1.0 - tau11 * t[1,1]
    for k = 2:n
        w[k,1] = - tau11 * t[1,k]
    end
    for k = 1:n
        w[n+k,1] = - tau12 * t[1,k]
    end
    for j = 2:n
        w[j-1,j] = - tau11 * t[j,j-1]
        w[j,j] = 1.0 - tau11 * t[j,j]
        for k = j+1:n
            w[k,j] = - tau11 * t[j,k]
        end
        for k = j-1:n
            w[n+k,j] = - tau12 * t[j,k]
        end
    end
    for k = 1:n
        w[k,n+1] = - tau21 * t[1,k]
    end
    w[n+1,n+1] = 1.0 - tau22 * t[1,1]
    for k = 2:n
        w[n+k,n+1] = - tau22 * t[1,k]
    end
    for j = 2:n
        for k = j-1:n
            w[k,n+j] = - tau21 * t[j,k]
        end
        w[n+j-1,n+j] = - tau22 * t[j,j-1]
        w[n+j,n+j] = 1.0 - tau22 * t[j,j]
        for k = j+1:n
            w[n+k,n+j] = - tau22 * t[j,k]
        end
    end

    for k = 1:n
        b2[k] = b[i-1,k]
        b2[n+k] = b[i,k]
    end
    linsolve_core!(linsolve_ws,Ref{UInt8}('T'),w,b2)
    for k = 1:n
        b[i-1,k] = b2[k]
        b[i,k] = b2[n+k]
    end

    b2n = copy(b[i-1:i,:])
    A_mul_Bt!(b2n,t)
    for k = 1:n
        for j = 1:i-2
            b[j,k] += t[j,i-1]*b2n[1,k] + t[j,i]*b2n[2,k]
        end
    end
    b     
end


    
