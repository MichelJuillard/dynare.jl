include("dgges_algo.jl")
include("dgesvx_algo.jl")

import Base.LinAlg.BLAS: scal!

type GsSolverWS
    dgges_ws::DggesWS
    dgesvx_ws::DgesvxWS
    Z11::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},1}
    Z12::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},1}
    Z21::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},1}
    Z22::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},1}
    tmp1::Array{Float64,2}
    gx::Array{Float64,2}
    
    function GsSolverWS(model,D,E)
        dgges_ws = DggesWS(Ref{UInt8}('N'),Ref{UInt8}('V'),E,D)
        n_dyn = model.n_dyn
        ns = model.n_bkwrd + model.n_both
        dgesvx_ws = DgesvxWS(n_dyn-ns,ns)
        Z11 = sub(dgges_ws.vsr,1:ns,1:ns)
        Z12 = sub(dgges_ws.vsr,1:ns,ns+1:n_dyn)
        Z21 = sub(dgges_ws.vsr,ns+1:n_dyn,1:ns)
        Z22 = sub(dgges_ws.vsr,ns+1:n_dyn,ns+1:n_dyn)
        tmp1 = Array(Float64,n_dyn-ns,ns)
        gx = Array(Float64,n_dyn-ns,ns)
        new(dgges_ws,dgesvx_ws,Z11,Z12,Z21,Z22,tmp1,gx)
    end
end

function gs_solver_core!(ws,D,E,model,qz_criterium,check=false)
    
    dgges_core!(ws.dgges_ws,E,D)
    ns = ws.dgges_ws.sdim[]
    if ns != model.n_bkwrd + model.n_both
        error("BK conditions aren't met")
    end
    ws.tmp1 = ws.Z12'
    gx = -(ws.Z12/ws.Z22)'
    dgesvx_core!(ws.dgesvx_ws,Ref{UInt8}('E'),Ref{UInt8}('T'),ws.Z22,ws.tmp1,ws.gx)
    scal!(length(ws.gx),-1.0,ws.gx,1)
    hx1 = ws.dgges_ws.vsr[1:ns,1:ns]/D[1:ns, 1:ns]
    hx2 = E[1:ns,1:ns]/ws.dgges_ws.vsr[1:ns,1:ns]
    hx = hx1*hx2
    ghx = zeros(model.endo_nbr, model.n_bkwrd + model.n_both)
    ghx[model.i_bkwrd,:] = hx[1:model.n_bkwrd,:]
    ghx[model.i_fwrd,:] = ws.gx[1:model.n_fwrd,:]
    ghx[model.i_both,:] = ws.gx[model.n_fwrd+(1:model.n_both),:]
    ghx, ws.gx, hx
end
