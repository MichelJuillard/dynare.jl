###
# solving (I - s^T ⊗ s^T ⊗ ... \otimes s^T \otimes t)x = d
###
include("quasi_upper_triangular.jl")
include("kronecker_utils.jl")

using linsolve_algo

import Base.LinAlg.BLAS: gemm!
export general_sylvester_solver!, real_eliminate!, solvi, transformation1

function general_sylvester_solver!(a,b,c,d,order)
    b = a\b
    d = a\d
    F = schur_zero(b)
    G = similarity_decomposition(c)
    T = F[:T]
    Q = F[:Z]
    S = G[:T]
    Z = G[:Z]
    D = Q*d
    D = multkron(D,Z',order)
    solver!(T,S,D,order)
    D = Q'D
    D = multkron(D,Z,order)
    d = D
end

function schur_zero(a)
    F = schurfact(a)
end

function similarity_decomposition(a)
    F = schurfact(a)
end

function multkron(a,b,k)
    c = b
    for i=1:k-1
        c = kron(c,b)
    end
    c = a*c
end

function solver!(t,s,d,order)
    n = size(t,1)
    s2 = s*s
    t2 = QuasiUpperTriangular(t*t)
    td = similar(d)
    solvi(1.0,order,t,t2,s,s2,d,td)
    d
end

function solvi(r,depth,t,t2,s,s2,d,td)
    n = size(t,2)
    if depth == 0
        copy!(d,(eye(length(d)) + r*t)\d)
        copy!(td,t*d)
    else
        nd = n^depth
        nd2 = 2*nd
        drange1 = 1:nd
        drange2 = 1:nd2
        i = 1
        while i <= n
            # s is transposed
            if i == n || s[i+1,i] == 0
                dv = view(d,drange1)
                tdv = view(td,drange1)
                solvi(r*s[i,i],depth-1,t,t2,s,s2,dv,tdv)
                if i < n
                    solvi_real_eliminate!(i,n,nd,drange1,depth-1,r,t,s,d,tdv)
                end
                drange1 += nd
                drange2 += nd
                i += 1
            else
                dv = view(d,drange2)
                tdv = view(td,drange2)
                # s transposed!
                solvii(r*s[i,i],r*s[i+1,i],r*s[i,i+1],depth-1,t,t2,s,s2,dv,tdv)
                if i < n - 1
                    solvi_complex_eliminate!(i,n,nd,drange1,depth-1,r,t,s,d,tdv)
                end
                drange1 += nd2
                drange2 += nd2
                i += 2
            end
        end
    end
end

function solvi_real_eliminate!(i,n,nd,drange,depth,r,t,s,d,td)
    d_copy = copy(d)
    d1 = view(d_copy,drange)
    mult_level!(depth,0,t,d1)
    for k = 1:depth
        mult_level_t!(depth-k,k,s,d1)
    end
    for j = i+1:n
        drange += nd
        dv = view(d,drange)
        # d += -r*s[i,j]*(s^T ⊗  s^T... ⊗ s^T)*td
        # can probably be optimized
        for k = 1:nd
            dv[k] -= r*s[i,j]*d1[k]
        end
    end 
end

function solvi_complex_eliminate!(i,n,nd,drange,depth,r,t,s,d,tdv)
    d_copy = copy(d)
    x1 = view(d_copy,drange)
    drange += nd
    x2 = view(d_copy,drange)
    mult_level!(depth,0,t,x1)
    mult_level!(depth,0,t,x2)
    for k = 1:depth
        mult_level_t!(depth-k,k,s,x1)
        mult_level_t!(depth-k,k,s,x2)
    end
    for j = i+2:n
        drange += nd
        dv = view(d,drange)
        # can probably be optimized
        for k = 1:nd
            dv[k] -= r*(s[i,j]*x1[k] + s[i+1,j]*x2[k])
        end
    end 
end

function solvii(alpha,beta1,beta2,depth,t,t2,s,s2,d,td)
    n = size(t,2)
    nd = n^(depth+1)
    transformation1(alpha,beta1,beta2,depth,t,s,d)
    dv = view(d,1:nd)
    solviip(alpha,sqrt(-beta1*beta2),depth,t,t2,s,s2,dv,td)
    dv = view(d,nd+1:2*nd)
    solviip(alpha,sqrt(-beta1*beta2),depth,t,t2,s,s2,dv,td)
end

function transformation1(a,b1,b2,depth,t,s,d)
    n = size(t,2)
    nd = n^(depth+1)
    d_orig = copy(d)
    drange = 1:nd
    d1 = view(d_orig,drange)
    d2 = view(d_orig,drange+nd)
    mult_level!(depth,0,t,d1)
    mult_level!(depth,0,t,d2)
    for i=1:depth
        mult_level_t!(depth-i,i,s,d1)
        mult_level_t!(depth-i,i,s,d2)
    end
    for i = drange
        d[i] += a*d1[i] - b1*d2[i]
        d[i+nd] += -b2*d1[i] + a*d2[i] 
    end
end

diag_zero_sq = 1e-30

function solviip(alpha,beta,depth,t,t2,s,s2,d,td)
    n = size(t,2)
    if beta*beta < diag_zero_sq
        solvi(alpha,depth,t,t2,s,s2,d,td)
        solvi(alpha,depth,t,t2,s,s2,d,td)
        return
    end

    if depth == 0
        tt = 2*alpha*t + (alpha*alpha+beta*beta)*t2
        copy!(d,(eye(length(d)) + r*tt)\d)
        copy!(td,tt*d)
    else
        nd = n^depth
        nd2 = 2*nd
        drange1 = 1:nd
        drange2 = 1:nd2
        i = 1
        while i <= n
            # s is transposed
            if i == n || s[i+1,i] == 0
                dv = view(d,drange1)
                tdv = view(td,drange1)
                if s[i,i]*s[i,i]*(alpha*alpha+beta*beta) > diag_zero_sq
                    solviip(s[i,i]*alpha,s[i,i]*beta,depth-1,t,t2,s,s2,dv,tdv)
                end
                if i < n
                    solviip_real_eliminate!(i,n,nd,drange1,depth-1,alpha,beta,t,t2,s,s2,d,tdv)
                end
                drange1 += nd
                drange2 += nd
                i += 1
            else
                dv = view(d,drange2)
                tdv = view(td,drange2)
                # s transposed !
                solviip2(alpha,beta,s[i,i],s[i+1,i],s[i,i+1],depth,t,t2,s,s2,dv,tdv)
                if i < n - 1
                    solviip_complex_eliminate!(i,n,nd,drange1,depth-1,alpha,beta,t,t2,s,s2,d,tdv)
                end
                drange1 += nd2
                drange2 += nd2
                i += 2
            end
        end
    end
end

function solviip_real_eliminate!(i,n,nd,drange,depth,alpha,beta,t,t2,s,s2,d,tdv)
    y1 = copy(d[drange])
    y2 = copy(d[drange])
    mult_level!(depth,0,t,y1)
    mult_level!(depth,0,t2,y2)
    for k = 1:depth
        mult_level_t!(depth-k,k,s,y1)
        mult_level_t!(depth-k,k,s2,y2)
    end
    for j = i+1:n
        drange += nd
        dv = view(d,drange)
        for k = 1:nd
            dv[k] -= 2*alpha*s[i,j]*y1[k] + (alpha*alpha+beta*beta)*s2[i,j]*y2[k]
        end
    end 
end

function solviip2(alpha,beta,gamma,delta1,delta2,depth,t,t2,s,s2,d,td)
    aspds = alpha*alpha + beta*beta
    gspds = gamma*gamma - delta1*delta2
    nd = n^depth
    dv1 = view(d,1:nd)
    dv2 = view(d,nd+1:2*nd)
    if aspds*gspds > diag_zero_sq
        transform2(alpha, beta, gamma, -delta1, -delta2, nd, depth, t, t2, s, s2, d)
        delta = sqrt(-delta1*delta2)
	a1 = alpha*gamma - beta*delta
	b1 = alpha*delta + gamma*beta
	a2 = alpha*gamma + beta*delta
	b2 = alpha*delta - gamma*beta
	solviip(a2, b2, depth-1, t, t2, s, s2, dv1, td);
	solviip(a1, b1, depth-1, t, t2, s, s2, dv1, td);
	solviip(a2, b2, depth-1, t, t2, s, s2, dv2, td);
	solviip(a1, b1, depth-1, t, t2, s, s2, dv2, td);
    end
end

function transform2(alpha, beta, gamma, delta1, delta2, nd, depth, t, t2, s, s2, d)
    drange = 1:nd
    d1 = view(d,drange)
    d2 = view(d,drange+nd)
    d1tmp = copy(d1)
    d2tmp = copy(d2)
    x1 = copy(d1);
    x2 = copy(d2);
    mult_level!(depth-1,0,t,d1tmp)
    mult_level!(depth-1,0,t,d2tmp)
    for i=1:depth-1
        mult_level_t!(depth-i-1,i,s,d1tmp)
        mult_level_t!(depth-i-1,i,s,d2tmp)
    end
    
    for i = 1:nd
        x1[i] += 2*alpha*(gamma*d1tmp[i] + delta1*d2tmp[i])
        x2[i] += 2*alpha*(delta2*d1tmp[i] + gamma*d2tmp[i])
    end
    
    d1tmp = copy(d1); # restore to d1
    d2tmp = copy(d2); # restore to d2
    mult_level!(depth-1,0,t2,d1tmp)
    mult_level!(depth-1,0,t2,d2tmp)
    for i=1:depth-1
        mult_level_t!(depth-i-1,i,s2,d1tmp)
        mult_level_t!(depth-i-1,i,s2,d2tmp)
    end

    aspds = alpha*alpha + beta*beta;
    gspds = gamma*gamma + delta1*delta2;
    for i = 1:nd
        x1[i] += aspds*(gspds*d1tmp[i] + 2*gamma*delta1*d2tmp[i])
        x2[i] += aspds*(2*gamma*delta2*d1tmp[i] + gspds*d2tmp[i])
    end
    
    copy!(d1,x1)
    copy!(d2,x2)
end

"""
    solviip_complex_eliminate!(i,n,nd,drange,depth,alpha,beta,t,t2,s,s2,d,tdv)

perfoms elimination after solving for a complex diagonal block of size 2*n^depth

d n^(depth+2) x 1

The solution is stored in d[drange; drange + nd]

The function updates d[i*nd+1:n*nd]
"""
function solviip_complex_eliminate!(i,n,nd,drange,depth,alpha,beta,t,t2,s,s2,d,tdv)
    @assert length(d) == n^(depth+2) [length(d), n^(depth+2)]
    @assert nd == n^(depth+1) [nd, n^(depth+1)]
    @assert length(drange) == nd [length(drange), nd]
    y11 = copy(d[drange])
    y12 = copy(d[drange])
    drange += nd
    y21 = copy(d[drange])
    y22 = copy(d[drange])
    mult_level!(depth,0,t,y11)
    mult_level!(depth,0,t2,y12)
    mult_level!(depth,0,t,y21)
    mult_level!(depth,0,t2,y22)
    for k = 1:depth
        mult_level_t!(depth-k,k,s,y11)
        mult_level_t!(depth-k,k,s2,y12)
        mult_level_t!(depth-k,k,s,y21)
        mult_level_t!(depth-k,k,s2,y22)
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
