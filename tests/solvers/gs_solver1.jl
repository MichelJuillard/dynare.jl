include("dgges.jl")
using DGGES: dgges!


function gs_solver1(D,E,model,qz_criterium,check=false)
    D,E,e,ns,Q,Z = dgges!('N','V',E,D)
    gx = -(Z[1:ns,ns+1:end]/Z[ns+1:end,ns+1:end])'
    hx1 = Z[1:ns,1:ns]/E[1:ns, 1:ns]
    hx2 = D[1:ns,1:ns]/Z[1:ns,1:ns]
    hx = hx1*hx2
    ghx = zeros(model.endo_nbr, model.n_bkwrd + model.n_both)
    ghx[model.i_bkwrd,:] = hx[1:model.n_bkwrd,:]
    ghx[model.i_fwrd,:] = gx[1:model.n_fwrd,:]
    ghx[model.i_both,:] = gx[model.n_fwrd+(1:model.n_both),:]
    ghx, gx, hx
end
