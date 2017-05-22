###
# solving (I - s^T ⊗ s^T ⊗ ... \otimes s^T \otimes t)x = d
###
include("quasi_upper_triangular.jl")
include("kronecker_utils.jl")

using linsolve_algo
using Schur

import Base.LinAlg.BLAS: gemm!
export EyePlusAtKronBWS, general_sylvester_solver!, real_eliminate!, solvi, transformation1

immutable EyePlusAtKronBWS
    n_s::Int64
    n_t::Int64
    vs_b::Array{Float64,2}
    vs_c::Array{Float64,2}
    s2::QuasiUpperTriangular{Float64,Matrix{Float64}}
    t2::QuasiUpperTriangular{Float64,Matrix{Float64}}
    work1::Array{Float64,1}
    work2::Array{Float64,1}
    work3::Array{Float64,1}
    linsolve_ws::LinSolveWS
    dgees_ws_b::DgeesWS
    dgees_ws_c::DgeesWS
    function EyePlusAtKronBWS(a::AbstractArray,b::AbstractArray,order::Int64,c::AbstractArray)
        n_t, n_s = LinAlg.checksquare(a, b)
        mc, nc = size(c)
        if mc != n_t
            DimensionMismatch("a has dimensions ($n_t, $n_t) but c has dimensions ($mc, $nc)")
        elseif nc != nc^order
            DimensionMismatch("b has dimensions ($n_s, $n_s), order is $order, but c has dimensions  ($mc, $nc) and $n_s^$order = $(n_s^order)")
        end
        vs_b = Matrix{Float64}(n_t,n_t)
        vs_c = Matrix{Float64}(nc,nc)
        s2 = QuasiUpperTriangular(Matrix{Float64}(nc,nc))
        t2 = QuasiUpperTriangular(Matrix{Float64}(n_t,n_t))
        linsolve_ws = LinSolveWS(n_s)
        dgees_ws_b = DgeesWS(b)
        dgees_ws_c = DgeesWS(c)
        work1 = Vector{Float64}(n_t*nc^order)
        work2 = Vector{Float64}(n_t*nc^order)
        work3 = Vector{Float64}(n_t*nc^order)
        new(n_s,n_t,vs_b,vs_c,s2,t2,work1,work2,work3,linsolve_ws,dgees_ws_b,dgees_ws_c)
    end
end

function swap(a,b)
    tmp = a
    b = a
    a = tmp
end
using Base.Test
function general_sylvester_solver!(a,b,c,d,order,ws)
    addr1 = pointer_from_objref(d)
    x = (kron(eye(size(c,2)^order),a) + kron(kron(c',c'),b))\vec(d)
    x = reshape(x,size(a,2),size(c,1)^order)
    display(x)
    @test a*x + b*x*kron(c,c) ≈ d
    a_orig = copy(a)
    b_orig = copy(b)
    d_orig = copy(d)
    linsolve_core!(ws.linsolve_ws,Ref{UInt8}('N'),a,b,d)
    @test b ≈ a_orig\b_orig
    @test d ≈ a_orig\d_orig
    @test x + b*x*kron(c,c) ≈ d
    dgees!(ws.dgees_ws_b,b)
    dgees!(ws.dgees_ws_c,c)
    t = QuasiUpperTriangular(b)
    A_mul_B!(ws.t2,t,t)
    s = QuasiUpperTriangular(c)
    A_mul_B!(ws.s2,s,s)
    result = reshape(ws.work1,size(t,2),size(s,1)^order)
    # do transpose in function
    d_orig = copy(d)
    a_mul_b_kron_c!(result,ws.dgees_ws_b.vs',d,ws.dgees_ws_c.vs,order,ws.work2)
    copy!(d,ws.work1)
    @test d ≈ ws.dgees_ws_b.vs'*d_orig*kron(ws.dgees_ws_c.vs,ws.dgees_ws_c.vs)
    QC = kron(ws.dgees_ws_c.vs,ws.dgees_ws_c.vs)
    @test ws.dgees_ws_b.vs'*x*QC + t*ws.dgees_ws_b.vs'*x*QC*kron(s,s) ≈ d
    d_vec = vec(d)
    d_orig = copy(d)
    solve1!(1.0,order,t,ws.t2,s,ws.s2,d_vec,ws)
    @test d + t*d*kron(s,s) ≈ d_orig
    # do transpose in function
    a_mul_b_kron_c!(result,ws.dgees_ws_b.vs,d,ws.dgees_ws_c.vs',order,ws.work2)
    copy!(d,result)
    @test d ≈ x
    println(addr1," ",pointer_from_objref(d))
end


function solver!(t::QuasiUpperTriangular,s::QuasiUpperTriangular,d::AbstractVector,order::Int64,ws::EyePlusAtKronBWS)
    s2 = QuasiUpperTriangular(s*s)
    t2 = QuasiUpperTriangular(t*t)
    solve1!(1.0,order,t,t2,s,s2,d,ws)
    d
end
using Base.Test
function solve1!(r,depth,t,t2,s,s2,d,ws)
    m = size(t,2)
    n = size(s,1)
    if depth == 0
        I_plus_rA_ldiv_B!(r,t,d)
    else
        nd = m*n^(depth-1)
        nd2 = 2*nd
        drange1 = 1:nd
        drange2 = 1:nd2
        i = 1
        while i <= n
            println("depth $(depth-1)")
#            T = t
#            for k = 1:depth-1
#                T = kron(s',T)
#            end
            if i == n || s[i+1,i] == 0
                println("solve1 branch 1")
                dv = view(d,drange1)
#                dv_orig = copy(dv)
                solve1!(r*s[i,i],depth-1,t,t2,s,s2,dv,ws)
#                @test dv ≈ (eye(n^depth) + r*s[i,i]*T)\dv_orig
                if i < n
                    solvi_real_eliminate!(i,n,nd,drange1,depth-1,r,t,s,d,ws)
                end
                drange1 += nd
                drange2 += nd
                i += 1
            else
                println("solve1 branch 2")
                dv = view(d,drange2)
                # s is transposed!
#                dv_orig = copy(dv)
                solvii(r*s[i,i],r*s[i+1,i],r*s[i,i+1],depth-1,t,t2,s,s2,dv,ws)
#                @test dv ≈ (eye(2*n^depth) + kron(r*s[i:i+1,i:i+1]',T))\dv_orig
                if i < n - 1
                    solvi_complex_eliminate!(i,n,nd,drange1,depth-1,r,t,s,d,ws)
                end
                drange1 += nd2
                drange2 += nd2
                i += 2
            end
        end
    end
end
using Base.Test
function solvi_real_eliminate!(i,n,nd,drange,depth,r,t,s,d,ws)
    d_copy = view(ws.work1,1:length(d))
    copy!(d_copy,d)
    d1 = view(d_copy,drange)
    work = view(ws.work2,1:length(drange))
    kron_at_kron_b_mul_c!(s,depth,t,d1,work)
    for j = i+1:n
        drange += nd
        dv = view(d,drange)
        for k = 1:nd
            dv[k] -= r*s[i,j]*d1[k]
        end
    end 
end

function solvi_complex_eliminate!(i,n,nd,drange,depth,r,t,s,d,ws)
    d_copy = view(ws.work1,1:length(d))
    d_copy = copy(d)
    work =  view(ws.work2,1:length(drange))
    x1 = view(d_copy,drange)
    drange += nd
    x2 = view(d_copy,drange)
    kron_at_kron_b_mul_c!(s,depth,t,x1,work)
    kron_at_kron_b_mul_c!(s,depth,t,x2,work)
    for j = i+2:n
        drange += nd
        dv = view(d,drange)
        # can probably be optimized
        for k = 1:nd
            dv[k] -= r*(s[i,j]*x1[k] + s[i+1,j]*x2[k])
        end
    end 
end
using Base.Test
function solvii(alpha,beta1,beta2,depth,t,t2,s,s2,d,ws)
    m = size(t,2)
    n = size(s,1)
    nd = m*n^depth
    d_orig = copy(d)
    transformation1(alpha,beta1,beta2,depth,t,s,d,ws)
#    d_target = d_orig + kron([alpha -beta1; -beta2 alpha],t)*d_orig
#    @test d ≈ d_target
    dv = view(d,1:nd)
#    dv_orig = copy(dv)
    solviip(alpha,sqrt(-beta1*beta2),depth,t,t2,s,s2,dv,ws)
#    @test dv ≈ (eye(n) + 2*alpha*t.data + (alpha*alpha - beta1*beta2)*t2.data)\dv_orig
    dv = view(d,nd+1:2*nd)
#    dv_orig = copy(dv)
    solviip(alpha,sqrt(-beta1*beta2),depth,t,t2,s,s2,dv,ws)
#    @test dv ≈ (eye(n) + 2*alpha*t.data + (alpha*alpha - beta1*beta2)*t2.data)\dv_orig
end

function transformation1(a,b1,b2,depth,t,s,d,ws)
    m = size(t,2)
    n = size(s,1)
    nd = m*n^depth
    d_orig = view(ws.work1,1:length(d))
    copy!(d_orig,d)
    drange = 1:nd
    d1 = view(d_orig,drange)
    d2 = view(d_orig,drange+nd)
    work = view(ws.work2,drange)
    kron_at_kron_b_mul_c!(s,depth,t,d1,work)
    kron_at_kron_b_mul_c!(s,depth,t,d2,work)
    for i = drange
        d[i] += a*d1[i] - b1*d2[i]
        d[i+nd] += -b2*d1[i] + a*d2[i] 
    end
end

diag_zero_sq = 1e-30

function solviip(alpha,beta,depth,t,t2,s,s2,d,ws)
    println("solviip depth $depth")
    m = size(t,2)
    n = size(s,1)
    if beta*beta < diag_zero_sq
        println("short cut")
        solve1!(alpha,depth,t,t2,s,s2,d,ws)
        solve1!(alpha,depth,t,t2,s,s2,d,ws)
        return
    end

    if depth == 0
        I_plus_rA_plus_sB_ldiv_C!(2*alpha,alpha*alpha+beta*beta,t,t2,d)
    else
        nd = m*n^(depth-1)
        nd2 = 2*nd
        drange1 = 1:nd
        drange2 = 1:nd2
        i = 1
        while i <= n
            # s is transposed
            if i == n || s[i+1,i] == 0
                println("branch 1")
                dv = view(d,drange1)
                println(s[i,i]*s[i,i]*(alpha*alpha+beta*beta))
                if s[i,i]*s[i,i]*(alpha*alpha+beta*beta) > diag_zero_sq
                    solviip(s[i,i]*alpha,s[i,i]*beta,depth-1,t,t2,s,s2,dv,ws)
                end
                if i < n
                    println("solviip_real_eliminate")
                    solviip_real_eliminate!(i,n,nd,drange1,depth-1,alpha,beta,t,t2,s,s2,d,ws)
                end
                drange1 += nd
                drange2 += nd
                i += 1
            else
                println("branch 2")
                dv = view(d,drange2)
                # s transposed !
                solviip2(alpha,beta,s[i,i],s[i+1,i],s[i,i+1],depth,t,t2,s,s2,dv,ws)
                if i < n - 1
                    println("solviip_complex_eliminate")
                    solviip_complex_eliminate!(i,n,nd,drange1,depth-1,alpha,beta,t,t2,s,s2,d,ws)
                end
                drange1 += nd2
                drange2 += nd2
                i += 2
            end
        end
    end
end

function solviip_real_eliminate!(i,n,nd,drange,depth,alpha,beta,t,t2,s,s2,d,ws)
    y1 = view(ws.work1,drange)
    y2 = view(ws.work1,drange+nd)
    copy!(y1,d[drange])
    copy!(y2,d[drange])
    work = view(ws.work2,1:length(drange))
    kron_at_kron_b_mul_c!(s,depth,t,y1,work)
    kron_at_kron_b_mul_c!(s2,depth,t2,y2,work)
    for j = i+1:n
        drange += nd
        dv = view(d,drange)
        for k = 1:nd
            dv[k] -= 2*alpha*s[i,j]*y1[k] + (alpha*alpha+beta*beta)*s2[i,j]*y2[k]
        end
    end 
end

function solviip2(alpha,beta,gamma,delta1,delta2,depth,t,t2,s,s2,d,ws)
    m = size(t,2)
    n = size(s,1)
    aspds = alpha*alpha + beta*beta
    gspds = gamma*gamma - delta1*delta2
    nd = m*n^(depth-1)
    dv1 = view(d,1:nd)
    dv2 = view(d,nd+1:2*nd)
    if aspds*gspds > diag_zero_sq
        transform2(alpha, beta, gamma, -delta1, -delta2, nd, depth, t, t2, s, s2, d,ws)
        delta = sqrt(-delta1*delta2)
	a1 = alpha*gamma - beta*delta
	b1 = alpha*delta + gamma*beta
	a2 = alpha*gamma + beta*delta
	b2 = alpha*delta - gamma*beta
	solviip(a2, b2, depth-1, t, t2, s, s2, dv1,ws);
	solviip(a1, b1, depth-1, t, t2, s, s2, dv1,ws);
	solviip(a2, b2, depth-1, t, t2, s, s2, dv2,ws);
	solviip(a1, b1, depth-1, t, t2, s, s2, dv2,ws);
    end
end

function transform2(alpha, beta, gamma, delta1, delta2, nd, depth, t, t2, s, s2, d, ws)
    drange = 1:nd
    d1 = view(d,drange)
    d2 = view(d,drange+nd)
    d1tmp = view(ws.work1,drange)
    d2tmp = view(ws.work1,drange+nd)
    copy!(d1tmp,d1)
    copy!(d2tmp,d2)
    x1 = view(ws.work2,drange)
    x2 = view(ws.work2,drange+nd)
    copy!(x1,d1);
    copy!(x2,d2);
    work = view(ws.work3,1:length(drange))
    kron_at_kron_b_mul_c!(s,depth-1,t,d1tmp,work)
    kron_at_kron_b_mul_c!(s,depth-1,t,d2tmp,work)

    for i = 1:nd
        d1[i] += 2*alpha*(gamma*d1tmp[i] + delta1*d2tmp[i])
        d2[i] += 2*alpha*(delta2*d1tmp[i] + gamma*d2tmp[i])
    end
    
    copy!(d1tmp,x1); # restore to d1
    copy!(d2tmp,x2); # restore to d2
    kron_at_kron_b_mul_c!(s2,depth-1,t2,d1tmp,work)
    kron_at_kron_b_mul_c!(s2,depth-1,t2,d2tmp,work)

    aspds = alpha*alpha + beta*beta;
    gspds = gamma*gamma + delta1*delta2;
    for i = 1:nd
        d1[i] += aspds*(gspds*d1tmp[i] + 2*gamma*delta1*d2tmp[i])
        d2[i] += aspds*(2*gamma*delta2*d1tmp[i] + gspds*d2tmp[i])
    end
end
using Base.Test
"""
    solviip_complex_eliminate!(i,n,nd,drange,depth,alpha,beta,t,t2,s,s2,d)

perfoms elimination after solving for a complex diagonal block of size 2*n^depth

d n^(depth+2) x 1

The solution is stored in d[drange; drange + nd]

The function updates d[i*nd+1:n*nd]
"""
function solviip_complex_eliminate!(i,n,nd,drange,depth,alpha,beta,t,t2,s,s2,d,ws)
    y11 = view(ws.work1,drange)
    y12 = view(ws.work1,drange+nd)
    copy!(y11,d[drange])
    copy!(y12,d[drange])
    drange += nd
    y21 = view(ws.work2,drange)
    y22 = view(ws.work2,drange+nd)
    copy!(y21,d[drange])
    copy!(y22,d[drange])
    work = view(ws.work3,drange)
    y11_o = copy(y11)
    y12_o = copy(y12)
    y21_o = copy(y21)
    y22_o = copy(y22)
    kron_at_kron_b_mul_c!(s,depth,t,y11,work)
    kron_at_kron_b_mul_c!(s2,depth,t2,y12,work)
    kron_at_kron_b_mul_c!(s,depth,t,y21,work)
    kron_at_kron_b_mul_c!(s2,depth,t2,y22,work)

    if depth == 1
        @test y11 ≈ kron(s',t)*y11_o
    end
    for j = i+2:n
        drange += nd
        dv = view(d,drange)
        for k = 1:nd
            dv[k] -= (2*alpha*s[i,j]*y11[k] + (alpha*alpha+beta*beta)*s2[i,j]*y12[k]
                      + 2*alpha*s[i+1,j]*y21[k] + (alpha*alpha+beta*beta)*s2[i+1,j]*y22[k])
        end
    end 
end
