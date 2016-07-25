include("dgges_algo.jl")
using DGGES

type GsSolverWS
    dgges_ws::DGGES.DggesWS

    function GsSolverWS(D,E)
        dgges_ws = DGGES.DggesWS(Ref{UInt8}('N'),Ref{UInt8}('V'),E,D)
        new(dgges_ws)
    end
end

function gs_solver_core!(ws,D,E,model,qz_criterium,check=false)
    
    DGGES.dgges_core!(ws.dgges_ws,E,D)
    ns = ws.dgges_ws.sdim[]
    gx = -(ws.dgges_ws.vsr[1:ns,ns+1:end]/ws.dgges_ws.vsr[ns+1:end,ns+1:end])'
    hx1 = ws.dgges_ws.vsr[1:ns,1:ns]/E[1:ns, 1:ns]
    hx2 = D[1:ns,1:ns]/ws.dgges_ws.vsr[1:ns,1:ns]
    hx = hx1*hx2
    ghx = zeros(model.endo_nbr, model.n_bkwrd + model.n_both)
    ghx[model.i_bkwrd,:] = hx[1:model.n_bkwrd,:]
    ghx[model.i_fwrd,:] = gx[1:model.n_fwrd,:]
    ghx[model.i_both,:] = gx[model.n_fwrd+(1:model.n_both),:]
    ghx, gx, hx
end
