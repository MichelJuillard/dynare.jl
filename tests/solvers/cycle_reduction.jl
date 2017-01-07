module cycle_reduction

using model
using linsolve_algo

import Base.LinAlg.BLAS: scal!, gemm!

export CycleReductionWS, cycle_reduction_core!, get_ABC!

type CycleReductionWS
    linsolve_ws::LinSolveWS
    A::Array{Float64,2}
    B::Array{Float64,2}
    C::Array{Float64,2}
    Ahat1::Array{Float64,2}
    Bcopy::Array{Float64,2}
    tmp::Array{Float64,2}
    tmp_00::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    tmp_02::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    tmp_20::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    tmp_22::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    tmp1::Array{Float64,2}
    A0::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    A2::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    tmp2::Array{Float64,2}
    tmp2_1::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    tmp2_2::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    info::Array{Int64,1}

    function CycleReductionWS(n,A,B,C)
        linsolve_ws = LinSolveWS(n)
        Ahat1 = Array(Float64,n,n)
	Bcopy = Array(Float64,n,n)
        tmp = Array(Float64,2*n,2*n)
        tmp_00 = view(tmp,1:n,1:n)
        tmp_02 = view(tmp,1:n,n+(1:n))
        tmp_20 = view(tmp,n+(1:n),1:n)
        tmp_22 = view(tmp,n+(1:n),n+(1:n))
        tmp1 = Array(Float64,n,2*n)
        A0 = view(tmp1,1:n,1:n)
        A2 = view(tmp1,1:n,n+(1:n))
        tmp2 = Array(Float64,2*n,n)
        tmp2_1 = view(tmp2,1:n,1:n)
        tmp2_2 = view(tmp2,n+(1:n),1:n)
        info = [0;0]
        new(linsolve_ws,A,B,C,Ahat1,Bcopy,tmp, tmp_00, tmp_02, tmp_20, tmp_22, tmp1, A0, A2, tmp2, tmp2_1, tmp2_2, info) 
    end
end

function CycleReductionWS(n)
    A = zeros(Float64,n,n)
    B = zeros(Float64,n,n)
    C = zeros(Float64,n,n)
    CycleReductionWS(n,A,B,C)
end	

function get_ABC!(ws::CycleReductionWS,model::Model,jacobian::Array{Float64})
    i_rows = model.n_static+1:model.endo_nbr
    ws.A[:,model.i_bkwrd_ns] = view(jacobian,i_rows,model.p_bkwrd_b)
    ws.B[:,model.i_current]  = view(jacobian,i_rows,model.p_current)
    ws.C[:,model.i_fwrd_ns]  = view(jacobian,i_rows,model.p_fwrd_b)
end

function cycle_reduction_core!(ws::CycleReductionWS, cvg_tol::Float64, max_it::Int64)
    n = size(ws.A,1)
    it = 0

    copy!(ws.Ahat1,1,ws.B,1,length(ws.B))
    @inbounds ws.tmp1[:,1:n] = ws.A
    @inbounds ws.tmp1[:,n+(1:n)] = ws.C
    @inbounds ws.tmp2[1:n,:] = ws.A
    @inbounds ws.tmp2[n+(1:n),:] = ws.C
    while true
        #        ws.tmp = ([A0; A2]/A1)*[A0 A2]
	copy!(ws.Bcopy,ws.B)
        linsolve_core!(ws.linsolve_ws,Ref{UInt8}('N'),ws.Bcopy,ws.tmp1)
        gemm!('N','N',-1.0,ws.tmp2,ws.tmp1,0.0,ws.tmp)
        for j=1:n
            @simd for i=1:n
                @inbounds ws.B[i,j] += ws.tmp_02[i,j] + ws.tmp_20[i,j]
            end
        end
        @inbounds ws.tmp1[:,1:n] = ws.tmp_00
        @inbounds ws.tmp1[:,n+(1:n)] = ws.tmp_22
        @inbounds ws.tmp2[1:n,:] = ws.tmp_00
        @inbounds ws.tmp2[n+(1:n),:] = ws.tmp_22
        for j=1:n
            @simd for i=1:n
                @inbounds ws.Ahat1[i,j] += ws.tmp_20[i,j]
            end
        end
        crit = norm(ws.A0,1)
        if crit < cvg_tol
	   # keep iterating until condition on A2 is met
            if norm(ws.A2,1) < cvg_tol
                break
            end
        elseif isnan(crit) || it == max_it
            if crit < cvg_tol
                ws.info[1] = 4
                ws.info[2] = log(norm(A2,1))
            else
                ws.info[1] = 3
                ws.info[2] = log(norm(A1,1))
            end
            return
        end        
        it += 1
    end
    #(ws.Ahat1\ws.A
    linsolve_core!(ws.linsolve_ws,Ref{UInt8}('N'),ws.Ahat1,ws.A)
    scal!(length(ws.A),-1.0,ws.A,1)

end

function cycle_reduction_check(X::Array{Float64,2},A0::Array{Float64,2}, A1::Array{Float64,2}, A2::Array{Float64,2},cvg_tol::Float64)
    res = A0 + A1*X + A2*X*X
    if (sum(sum(abs.(res))) > cvg_tol)
        print("the norm of the residuals, ", res, ", compared to the tolerance criterion ",cvg_tol)
    end
    nothing
end
    
#function cycle_reduction(A0, A1, A2, cvg_tol = 1e-8, maxit = 300, check = false)
#    n,m = size(A0)
#    if check
#        A0_0 = A0
#        A1_0 = A1
#        A2_0 = A2
#    end
#    id0, id2, A0_0, Ahat1, tmp = cycle_reduction_init(n)
#    info = cycle_reduction_core(A0, A1, A2, cvg_tol, maxit, A0_0, Ahat1, tmp, id0, id2)
#    if check
#        cycle_reduction_check(A0,A0_0,A1_0,A2_0,cvg_tol)
#    end
#    return A0, info
#end

end