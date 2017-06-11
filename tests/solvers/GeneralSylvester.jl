module GeneralSylvester
include("quasi_upper_triangular.jl")
include("kronecker_utils.jl")

using linsolve_algo

import Base.LinAlg.BLAS: gemm!
export general_sylvester_solver!, real_eliminate!, solvi, transformation1

type GeneralSylvesterWS
    w1:: Vector{Float64}
    w2:: Vector{Float64}
    w3:: Vector{Float64}
    w4:: Vector{Float64}
    function GeneralSylvesterWS(n)
        w1 = Vector(n)
        w2 = similar(w1)
        w3:: similar(w1)
        w4:: similar(w1)
        new(w1, w2, w3, w4)
    end
end
    
function general_sylvester_solver!(a,b,c,d,order,ws::GeneralSylvesterWS)
    b = a\b
    d = a\d
    F = schur_zero(b)
    G = similarity_decomposition(c)
    T = F[:T]
    Q = F[:Z]
    S = G[:T]
    Z = G[:Z]
    w = Vector{Float64}(size(a,1)*size(c,2)^order)
    D = Matrix(size(a,1),size(c,2)^order)
    a_mul_b_kron_c!(D,Q,d,Z,order,w)
    solver!(T,S,D,order)
    D = a_mul_b_kron_c!(Q',D,Z',order,w)
    d = D
end

function schur_zero(a)
    F = schurfact(a)
end

function similarity_decomposition(a)
    F = schurfact(a)
end

function solver!(t,s,d,order,ws::GeneralSylvesterWS)
    n = size(t,1)
    index = zeros(order)
    depth = n
    index[depth] = 1
    solvi(1.0,index,depth,t,s,d,ws)
    d
end

"""
    function solvi(r,column_index,d_block_number,depth,t,s,d)
solves

    (I + r ((⊗^depth s^T)\otimes t)) x = d
"""   
function solvi(r,column_index,d_block_number,depth,t,s,d,ws::GeneralSylvesterWS)
    n = size(t,2)
    order = length(column_index)
    if depth == 0
        row_range = (d_block_number-1)*n + (1:n)
        d[row_range] = (eye(n) + r*t)\d[row_range]
        y = t*d[row_range]
        if row_range[1] > 1
            real_eliminate!(column_index,1:n,row_range[1],length(column_index),y,1.0,t,s,d)
        end
   else
        i = n
        while i > 0
            column_index[depth] = i
            if i == 1 || s[i,i-1] == 0
                solvi(r*s[i,i],column_index,d_block_number,depth-1,t,s,d,ws)
                d_block_number -= n^(depth-1)
                i -= 1
            else
                solvii(r*s[i-1:i,i-1:i],column_index,d_block_number,depth-1,t,s,d,ws)
                d_block_number -= 2*n^(depth-1)
                i -= 2
            end
        end
    end    
end

"""
    function real_eliminate!(column_index,row_range,row_max,depth,y,r,t,s,d)
performs elimination and updates d
    d_k -= r((⊗^depth s^T)\otimes t)d_k
"""
function real_eliminate!(column_index,row_range,row_max,depth,y,r,t,s,d)
    n = size(t,2)
    order = length(column_index)
    if depth == 0
        if abs(r) > eps()
            for (i, j) = enumerate(row_range)
                d[j] -= r*y[i]
            end
        end
    else
        for k=1:min(column_index[depth]+1,n)
            col = column_index[depth]
            real_eliminate!(column_index,row_range,row_max,depth-1,y,r*s[col,k],t,s,d)
            row_range += n^depth
            # stop at row_max
            if row_range[1] >= row_max
                break
            end
        end
    end
end

"""
    function solvii(r,column_index,d_block_number,depth,t,s,d)
solves

    (I + [r_11 r_12;r_21 r_22]^t⊗ ((⊗^i s^T)\otimes t)) x = d
"""   
function solvii(r,column_index,d_block_number,depth,t,s,d,ws::GeneralSylvesterWS)
    n = size(t,2)
    nd = n^(depth-1)
    row_range = (d_block_number-2)*nd + 1:2*nd
    dhat = transformation1(r[1,1],r[2,1],r[1,2],row_range,depth,t,s,d)
    row_range1 = row_range[1:nd]
    solviip(r[1,1]*r[2,2],r[1,2]*r[2,1],column_index,row_range1,depth,t,s,d,ws)
    row_range2 = row_range[nd+1:nd]
    solviip(r[1,1]*r[2,2],r[1,2]*r[2,1],column_index,row_range2,depth,t,s,d,ws)
end

function transformation1(a,b1,b2,row_range,depth,t,s,d,ws::GeneralSylvesterWS)
    n = size(t,2)
    nd_1 = n^depth
    j = 1:nd_1
    d1 = view(d,j)
    d2 = view(d,j+nd_1)
    kron_at_kron_b_mul_c!(s,depth-1,t,d1,w);
    kron_at_kron_b_mul_c!(s,depth-1,t,d2,w);
    for i=1:nd_1
        tmp = -b2*d1[i] + a*d2[i]
        d[i] += a*d1[i] - b1*d2[i]
        d[i+nd_1] += tmp
    end
end

diag_zero_sq = 1e-30
function solviip(r1,r2,column_index,row_range,depth,t,s,d,ws::GeneralSylvesterWS)
    # to be computed once only !!!
    t2 = t*t
    n = size(t,2)
    if abs(r2) < diag_zero_sq
        # to be checked !!!
        solvi(r1,column_index,d_block_number,depth,t,s,d,ws)
        solvi(r1,column_index,d_block_number,depth,t,s,d,ws)
    end

    if depth == 0
        row_range = (d_block_number-1)*n + (1:n)
        tt = 2*r1*t + (r1*r1+r2*r2)*t2
        d[row_range] = (eye(n) + tt)\d[row_range]
        y = t*d[row_range]
        if row_range[1] > 1
            real_eliminate!(column_index,1:n,row_range[1],length(column_index),y,1.0,t,s,d)
        end
   else
        
end

end
