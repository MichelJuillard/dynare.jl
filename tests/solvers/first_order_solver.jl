module FirstOrder

using model
using qr_algo
using CycleReduction
using gs_solver1
using linsolve_algo

import Base.LinAlg.BLAS: gemm!

type FirstOrderSolverWS
    jacobian_static::Array{Float64,2} 
    qr_ws::QrWS
    solver_ws::Union{GsSolverWS,CycleReductionWS}
    ghx::StridedMatrix{Float64}
    gx::Array{Float64,2}
    hx::Array{Float64,2}
    temp1::Array{Float64,2}
    temp2::Array{Float64,2}
    temp3::Array{Float64,2}
    temp4::Array{Float64,2}
    b10::Array{Float64,2}
    b11::Array{Float64,2}
    linsolve_ws::LinSolveWS
    
    function FirstOrderSolverWS(algo, jacobian, m)
        if m.n_static > 0
            jacobian_static = Array{Float64,2}(m.endo_nbr,m.n_static)
            qr_ws = QrWS(jacobian_static)
        else
            jacobian_static = Array(Float64,0,0)
            qr_ws = QrWS(Array(Float64,0,0))
        end
        if algo == "GS"
            d = zeros(m.n_dyn,m.n_dyn)
            e = zeros(m.n_dyn,m.n_dyn)
            solver_ws = GsSolverWS(d,e,m.n_bkwrd+m.n_both)
        elseif algo == "CR"
            n = m.endo_nbr - m.n_static
            solver_ws = CycleReductionWS(n)
        end
        ghx = Array(Float64,m.endo_nbr,m.n_bkwrd+m.n_both)
        gx = Array(Float64,m.n_fwrd+m.n_both,m.n_bkwrd+m.n_both)
        hx = Array(Float64,m.n_bkwrd+m.n_both,m.n_bkwrd+m.n_both)
        temp1 = Array(Float64,m.n_static,m.n_fwrd+m.n_both)
        temp2 = Array(Float64,m.n_static,m.n_bkwrd+m.n_both)
        temp3 = Array(Float64,m.n_static,m.n_bkwrd+m.n_both)
        temp4 = Array(Float64,m.endo_nbr - m.n_static,m.n_bkwrd+m.n_both)
        b10 = Array(Float64,m.n_static,m.n_static)
        b11 = Array(Float64,m.n_static,length(m.p_current))
        linsolve_ws = LinSolveWS(m.n_static)
        
        new(jacobian_static,qr_ws,solver_ws,ghx,gx,hx,temp1,temp2,temp3,temp4,b10,b11,linsolve_ws)
    end
end
        
function remove_static!(ws,jacobian,p_static)
    ws.jacobian_static[:,:] = view(jacobian,:,p_static)
    dgeqrf_core!(ws.qr_ws,ws.jacobian_static)
    dormrqf_core!(ws.qr_ws,Ref{UInt8}('L'),Ref{UInt8}('T'),ws.jacobian_static,
                      jacobian)
end

function add_static!(ws::FirstOrderSolverWS,jacobian::Array{Float64,2},model::Model)
    i_static = 1:model.n_static
    #    temp = - jacobian[i_static,model.p_fwrd_b]*gx*hx
    ws.temp1 = view(jacobian,i_static,model.p_fwrd_b)
    gemm!('N','N',1.0,ws.temp1,ws.gx,0.0,ws.temp2)
    gemm!('N','N',-1.0,ws.temp2,ws.hx,0.0,ws.temp3)
    ws.b10 = view(jacobian,i_static, model.p_static)
    ws.b11 = view(jacobian,i_static, model.p_current)
    ws.temp3 -= view(jacobian,i_static,model.p_bkwrd_b)
    for i=1:size(ws.ghx,2)
        for j=1:length(model.i_dyn)
            ws.temp4[j,i] = ws.ghx[model.i_dyn[j],i]
        end
    end
    gemm!('N','N',-1.0,ws.b11,ws.temp4,1.0,ws.temp3)
    linsolve_core!(ws.linsolve_ws,Ref{UInt8}('N'),ws.b10,ws.temp3)
    ws.ghx[model.i_static,:] = ws.temp3
end

function first_order_solver(ws,algo, jacobian, model, options)
    if model.n_static > 0
        remove_static!(ws,jacobian,model.p_static)
    end
    n = model.n_fwrd + model.n_bkwrd + model.n_both
    if algo == "CR"
        a, b, c = get_abc(model,jacobian)
	x = zeros(a)
        cycle_reduction!(x,a,b,c,ws.solver_ws,options.cycle_reduction.tol,100)
        if ws.solver_ws.info[1] > 0
            error("CR didn't converge")
        end
        ws.ghx[model.i_dyn,:] = view(x,:,model.i_bkwrd_ns)
        ws.gx[:,:] = view(ws.ghx,model.gx_rows,:)
        ws.hx[:,:] = view(ws.ghx,model.hx_rows,:)
    elseif algo == "GS"
        d, e = get_de(jacobian[model.n_static+1:end,:],model)
        gs_solver!(ws.solver_ws,d,e,model.n_bkwrd+model.n_both,options.generalized_schur.criterium)
        ws.gx = ws.solver_ws.g2
        ws.hx = ws.solver_ws.g1
        ws.ghx[model.i_bkwrd,:] = ws.solver_ws.g1[1:model.n_bkwrd,:]
        ws.ghx[model.i_fwrd,:] = ws.solver_ws.g2[1:model.n_fwrd,:]
        ws.ghx[model.i_both,:] = ws.solver_ws.g2[model.n_fwrd+(1:model.n_both),:]
    end
    if model.n_static > 0
        add_static!(ws,jacobian,model)
    end
end

end    
