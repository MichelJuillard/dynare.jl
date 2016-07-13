include("dgges.jl")
using DGGES: dgges!


function gs_solver1(D,E,qz_criterium,k1,k2,check=false)
    D,E,e,ns,Q,Z = dgges!('N','V',D,E)
    
    gx = -(Z[1:ns,ns+1:end]/Z[ns+1:end,ns+1:end])'
    hx1 = Z[1:ns,1:ns]/E[1:ns, 1:ns]
    hx2 = D[1:ns,1:ns]/Z[1:ns,1:ns]
    hx = hx1*hx2
    gx = [hx[k1,:]; gx[k2,:]]
end
