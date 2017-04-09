module GeneralSylvester
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
    index = zeros(order)
    depth = n
    index[depth] = 1
    solvi(1.0,index,depth,t,s,d)
    d
end

function solvi(r,column_index,row_range,depth,t,s,d)
    n = size(t,2)
    order = length(column_index)
    if depth == 0
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
                solvi(r*s[i,i],column_index,row_range,depth-1,t,s,d)
                row_range -= n^depth
                i -= 1
            else
                solvii(r*s[i-1:i,i-1:i],column_index,row_range,depth-1,t,s,d)
                row_range -= 2*n^depth
                i -= 2
            end
        end
    end    
end

function real_eliminate!(column_index,row_range,row_max,depth,y,r,t,s,d)
    n = size(t,2)
    order = length(column_index)
    if depth == 0
        println(row_range)
        if abs(r) > eps()
            println(row_range)
            for (i, j) = enumerate(row_range)
                d[j] -= r*y[i]
            end
        end
    else
        for k=1:min(column_index[depth]+1,n)
            col = column_index[depth]
            println("depth ",depth," k ",k," col ",col," row_max ",row_max," row_range ",row_range )
            real_eliminate!(column_index,row_range,row_max,depth-1,y,r*s[k,col],t,s,d)
            row_range += n^depth
            # stop at row_max
            if row_range[1] >= row_max
                break
            end
        end
    end
end

function solvii(r,index,index_j,depth,t,s,d)
    determinant = r[1,1]*r[2,2] - r[1,2]*r[2,1]
    if determinant > eps
        dhat = transformation1(r[1,1],r[1,2],r[2,1],depth,t,s,d)
        solviip(r[1,1]*r[2,2],r[1,2]*r[2,1],index,index_j,depth,t,s,d1)
        solviip(r[1,1]*r[2,2],r[1,2]*r[2,1],index,index_j,depth,t,s,d2)
    end
end

function transformation1(a,b1,b2,depth,t,s,d)
    n = size(t,2)
    nd_1 = n^depth
    j = 1:nd_1
    d0 = copy(d)
    d_orig = copy(d)
    d1 = view(d0,j)
    d2 = view(d0,j+nd_1)
    d1_orig = copy(d1)
    d2_orig = copy(d2)
    mult_kron!(t,s,d1,depth-1)
    mult_kron!(t,s,d2,depth-1)
    for i=1:nd_1
        d[i] += a*d1[i] - b1*d2[i]
        d[i+nd_1] += -b2*d1[i] + a*d2[i]
    end
end

end
