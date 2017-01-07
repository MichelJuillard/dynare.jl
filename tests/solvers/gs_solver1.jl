module gs_solver1

using dgges_algo
using linsolve_algo

import Base.LinAlg.BLAS: scal!, gemm!

export GsSolverWS, gs_solver_core!

type GsSolverWS
    dgges_ws::DggesWS
    linsolve_ws::LinSolveWS
    D11::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    E11::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    Z11::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    Z12::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    Z21::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    Z22::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    tmp1::Array{Float64,2}
    tmp2::Array{Float64,2}
    tmp3::Array{Float64,2}
    gx::Array{Float64,2}
    gx_f::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},Colon},false}
    hx::Array{Float64,2}
    ghx::Array{Float64,2}
    ghx_bb::SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Colon},false}
    ghx_f::SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Colon},false}

    function GsSolverWS(model,D,E)
        dgges_ws = DggesWS(Ref{UInt8}('N'),Ref{UInt8}('V'),E,D)
        n_dyn = model.n_dyn
        ns = model.n_bkwrd + model.n_both
        linsolve_ws = LinSolveWS(n_dyn-ns)
        D11 = view(D,1:ns,1:ns)
        E11 = view(E,1:ns,1:ns)
        Z11 = view(dgges_ws.vsr,1:ns,1:ns)
        Z12 = view(dgges_ws.vsr,1:ns,ns+1:n_dyn)
        Z21 = view(dgges_ws.vsr,ns+1:n_dyn,1:ns)
        Z22 = view(dgges_ws.vsr,ns+1:n_dyn,ns+1:n_dyn)
        tmp1 = Array(Float64,n_dyn-ns,ns)
        tmp2 = Array(Float64,ns,ns)
        tmp3 = Array(Float64,ns,ns)
        gx = Array(Float64,n_dyn-ns,ns)
        gx_f = view(gx,1:model.n_fwrd,:)
        hx = Array(Float64,ns,ns)
        ghx = Array(Float64,model.endo_nbr,ns)
        ghx_bb = view(ghx,[model.i_bkwrd; model.i_both],:)
        ghx_f = view(ghx,model.i_fwrd,:)
#        new(dgges_ws,linsolve_ws,D11,E11,Z11,Z12,Z21,Z22,tmp1,tmp2,tmp3,gx,gx_f)
        new(dgges_ws,linsolve_ws,D11,E11,Z11,Z12,Z21,Z22,tmp1,tmp2,tmp3,gx,gx_f,hx,ghx,ghx_bb,ghx_f)
    end
end

function gs_solver_core!(ws::GsSolverWS,D,E,model,qz_criterium,check=false)
    
    dgges_core!(ws.dgges_ws,E,D)
    ns = ws.dgges_ws.sdim[]
    if ns != model.n_bkwrd + model.n_both
        error("BK conditions aren't met")
    end
    ws.gx = ws.Z12'
#    ws.tmp2 = ws.Z22'
#    gx = -(ws.Z12/ws.Z22)'
    linsolve_core!(ws.linsolve_ws,Ref{UInt8}('T'),ws.Z22,ws.gx)
    scal!(length(ws.gx),-1.0,ws.gx,1)
    ws.tmp2 = ws.Z11'
#    hx1 = ws.dgges_ws.vsr[1:ns,1:ns]/D[1:ns, 1:ns]
    ws.D11 = view(D,1:ns,1:ns)
    linsolve_core!(ws.linsolve_ws,Ref{UInt8}('T'),ws.D11,ws.tmp2)
#    hx2 = E[1:ns,1:ns]/ws.dgges_ws.vsr[1:ns,1:ns]
    ws.E11 = view(E,1:ns,1:ns)
    ws.tmp3 = ws.E11'
    linsolve_core!(ws.linsolve_ws,Ref{UInt8}('T'),ws.Z11,ws.tmp3)
    #hx = ws.tmp2*ws.tmp3
    gemm!('T','T',1.0,ws.tmp2,ws.tmp3,0.0,ws.hx)
    if false
        for i = 1:ns
            for j = 1:model.n_bkwrd
                ws.ghx[model.i_bkwrd[j],i] = ws.hx[j,i]
            end
            for j = 1:model.n_fwrd
                ws.ghx[model.i_fwrd[j],i] = ws.gx[j,i]
            end
            for j = 1:model.n_both
                ws.ghx[model.i_both[j],i] = ws.hx[model.n_bkwrd+j,i]
            end
        end
    else
        ws.ghx[model.i_bkwrd,:] = ws.hx[1:model.n_bkwrd,:]
        ws.ghx[model.i_fwrd,:] = ws.gx[1:model.n_fwrd,:]
        ws.ghx[model.i_both,:] = ws.gx[model.n_fwrd+(1:model.n_both),:]
    end
end

end