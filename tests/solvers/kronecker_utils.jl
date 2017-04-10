"""
    mult_kron!(s::QuasiUpperTriangular, t::QuasiUpperTriangular, x::StridedVector, depth::Int64)

Performs (t⊗t⊗...⊗t⊗t)*x. The solution is returned in vector x. depth indicates the number of occurences of matrix t

We use (t⊗t⊗...⊗t⊗s)*x = (t⊗I)*(I⊗t⊗I)*...*(I⊗t⊗I)*(I⊗s)*x

"""
function mult_kron!(s,t,x,depth)
    n = size(s,1)
    mult_level!(depth,0,s,x)
    for p=1:depth
        mult_level!(depth-p,p,t,x)
    end
end
    
"""
    mult_level!(level::Int64, t::QuasiUpperTriangular, x::AbstractVector, depth::Int64)

Performs (I_p ⊗ t ⊗ I_q) 
"""
function mult_level!(p::Int64, q::Int64, t::QuasiUpperTriangular, x::AbstractVector)
    n = size(t,2)
    if p > 0 && q > 0
        # (I_n^p ⊗ t ⊗ I_n^q)*x = [(t ⊗ I_n^q)*x_1 (t ⊗ I_n^q)*x_2 ... (t ⊗ I_n^q)*x_n^p]
        nq = n^q
        nq1 = nq*n
        j = 1:nq1
        for i=1:n^p
            xi = view(x,j)
            # (t ⊗ I_n^q)*x = (x'*(t'⊗ I_n^q))' = vec(reshape(x',n^q,n)*t')
            xi = vec(A_mul_Bt!(reshape(xi,nq,n),t))
            j += nq1
        end
    elseif q == 0 && p > 0
        #  (I_n^p⊗t)*x = vec(t*[x_1 x_2 ... x_p])
        x = vec(A_mul_B!(t,reshape(x,n,n^p)))
    elseif q == 0 && p == 0
        # t*x
        A_mul_B!(t,x)
    else
        # (t⊗I_n^q)*x = (x'*(t'⊗ I_n^q))' = vec(reshape(x',n^q,n)*t')
        x = vec(A_mul_Bt!(reshape(x,n^q,n),t))
    end
end
        
    
