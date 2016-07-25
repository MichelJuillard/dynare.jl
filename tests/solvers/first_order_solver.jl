include("cycle_reduction.jl")
include("gs_solver1.jl")

type Model
    endo_nbr
    lead_lag_incidence
    n_static
    n_fwrd
    n_bkwrd
    n_both
    DErows1
    DErows2
    n_dyn
    i_static
    i_dyn
    i_bkwrd
    i_bkwrd_b
    i_bkwrd_ns
    i_current
    i_fwrd
    i_fwrd_b
    i_fwrd_ns
    i_both
    p_static
    p_bkwrd
    p_bkwrd_b
    p_current
    p_fwrd
    p_fwrd_b
    p_both_b
    p_both_f
    icolsD
    jcolsD
    icolsE
    jcolsE
    colsUD
    colsUE
    i_cur_fwrd
end

function Model(endo_nbr,lead_lag_incidence)
    i_static = find((lead_lag_incidence[1,:] .== 0) & (lead_lag_incidence[3,:] .== 0))
    p_static = squeeze(lead_lag_incidence[2,i_static],1)
    i_dyn = find((lead_lag_incidence[1,:] .> 0) | (lead_lag_incidence[3,:] .> 0))
    n_static = length(i_static)
    i_bkwrd = find((lead_lag_incidence[1,:] .> 0) & (lead_lag_incidence[3,:] .== 0))
    i_bkwrd_b = find((lead_lag_incidence[1,:] .> 0))
    i_bkwrd_ns = find(lead_lag_incidence[1,i_dyn] .> 0)
    p_bkwrd = squeeze(lead_lag_incidence[1,i_bkwrd],1)
    p_bkwrd_b = squeeze(lead_lag_incidence[1,i_bkwrd_b],1)
    n_bkwrd = length(i_bkwrd)
    i_fwrd = find((lead_lag_incidence[3,:] .> 0) & (lead_lag_incidence[1,:] .== 0)) 
    i_fwrd_b = find((lead_lag_incidence[3,:] .> 0)) 
    i_fwrd_ns = find(lead_lag_incidence[3,i_dyn] .> 0)
    p_fwrd = squeeze(lead_lag_incidence[3,i_fwrd],1)
    p_fwrd_b = squeeze(lead_lag_incidence[3,i_dyn[i_fwrd_ns]],1)
    n_fwrd = length(i_fwrd)
    i_both = find((lead_lag_incidence[1,:] .> 0) & (lead_lag_incidence[3,:] .> 0)) 
    p_both_b = squeeze(lead_lag_incidence[1,i_both],1)
    p_both_f = squeeze(lead_lag_incidence[3,i_both],1)
    n_both = length(i_both)
    i_cur_fwrd = find((lead_lag_incidence[2,:] .> 0) & (lead_lag_incidence[3,:] .> 0))
    p_cur_fwrd = squeeze(lead_lag_incidence[2,i_cur_fwrd],1)
    n_cur_fwrd = length(i_cur_fwrd)
    junk, i_cur_bkwrd, p_cur_bkwrd = findnz(lead_lag_incidence[2,i_bkwrd])
    n_cur_bkwrd = length(i_cur_bkwrd)
    i_current = find(lead_lag_incidence[2,i_dyn] .> 0 )
    p_current = squeeze(lead_lag_incidence[2,i_dyn[i_current]],1)
    icolsD = [1:n_cur_bkwrd; n_bkwrd+n_both+(1:(n_fwrd+n_both))]
    jcolsD = [p_cur_bkwrd; p_fwrd; p_both_f]
    # derivatives of current values of variables that are both
    # forward and backward are included in the E matrix
    icolsE = [1:(n_bkwrd+n_both); n_bkwrd+n_both+(1:(n_fwrd+n_both))]
    jcolsE = [p_bkwrd; p_both_b; p_cur_fwrd]
    colsUD = n_bkwrd+(1:n_both)
    colsUE = n_both + n_fwrd + colsUD
    n_dyn = endo_nbr - n_static + n_both
    DErows1 = 1:(n_dyn-n_both)
    DErows2 = (n_dyn-n_both)+(1:n_both)
    gx_rows = n_bkwrd + n_both + (1:(n_fwrd+n_both))
    hx_rows = 1:(n_bkwrd + n_both)          
    Model(endo_nbr,lead_lag_incidence,n_static,n_fwrd,n_bkwrd,n_both,DErows1,DErows2,
          n_dyn,i_static,i_dyn,i_bkwrd,i_bkwrd_b,i_bkwrd_ns,i_current,i_fwrd,i_fwrd_b,i_fwrd_ns,i_both,p_static,
          p_bkwrd,p_bkwrd_b,p_current,p_fwrd,p_fwrd_b,p_both_b,p_both_f,
          icolsD,jcolsD,icolsE,jcolsE,colsUD,colsUE,i_cur_fwrd)                  
end
    
type FirstOrderSolverWS
    gs_solver_ws::GsSolverWS

    function FirstOrderSolverWS(algo, jacobian, model)
        if model.n_static > 0
            Q, jacobian_ = remove_static(jacobian,model.p_static)
        end
        D, E = get_DE(jacobian_[model.n_static+1:end,:],model)
        gs_solver_ws = GsSolverWS(D,E)
        new(gs_solver_ws)
    end
end
        
function get_ABC!(jacobian,model,A,B,C)
    A[:,model.i_fwrd_ns] = jacobian[:,model.p_fwrd_b]
    B[:,model.i_current] = jacobian[:,model.p_current]
    C[:,model.i_bkwrd_ns] = jacobian[:,model.p_bkwrd_b]
    return A, B, C
end

function remove_static(jacobian,p_static)
    n = length(p_static)
    Q = qr(jacobian[:,p_static];thin=false)
    jacobian_ = Q[1]'*jacobian
    return Q[1], jacobian_
end

function get_DE(jacobian,model)
    n1 = size(model.DErows1,1)
    n2 = model.n_dyn - n1;
    D = zeros(model.n_dyn,model.n_dyn)
    E = zeros(model.n_dyn,model.n_dyn)
    D[1:n1,model.icolsD] = jacobian[:,model.jcolsD]
    E[1:n1,model.icolsE] = -jacobian[:,model.jcolsE]
    U = eye(n2)                                    
    D[model.DErows2,model.colsUD] = U
    E[model.DErows2,model.colsUE] = U
    return D, E
end

function add_static(ghx,gx,hx,A,B,C,jacobian,model)
    temp = - jacobian[1:model.n_static,model.p_fwrd_b]*gx*hx
    B10 = jacobian[1:model.n_static, model.p_static]
    B11 = jacobian[1:model.n_static, model.p_current]
    temp = temp - jacobian[1:model.n_static,model.p_bkwrd_b]
    temp = B10\(temp-B11*ghx[model.i_dyn,:])
    ghx[model.i_static,:] = temp
    return ghx
end

function first_order_solver(ws,algo, jacobian, model, options)
    model = Model(model.endo_nbr,model.lead_lag_incidence)
    if model.n_static > 0
        Q, jacobian_ = remove_static(jacobian,model.p_static)
    end
    n = model.n_fwrd + model.n_bkwrd + model.n_both
    A = zeros(n,n)
    B = zeros(n,n)
    C = zeros(n,n)
    A, B, C = get_ABC!(jacobian_[model.n_static+1:end,:],model,A,B,C)
    if algo == "CR"
        ghx = cycle_reduction(A,B,C,options.cycle_reduction.tol)
        gx = ghx(model.gx_rows,:)
        hx = ghx(model.hx_rows,:)
    elseif algo == "GS"
        D, E = get_DE(jacobian_[model.n_static+1:end,:],model)
        ghx,gx,hx = gs_solver_core!(ws.gs_solver_ws,D,E,model,options.generalized_schur.criterium)
    end
    if model.n_static > 0
        ghx = add_static(ghx,gx,hx,A,B,C,jacobian_,model)
    end
    ghx, gx, hx
end

    
