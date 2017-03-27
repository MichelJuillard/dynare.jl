module CycleReduction

using model
using linsolve_algo

import Base.LinAlg.BLAS: scal!, gemm!

export CycleReductionWS, cycle_reduction_core!

type CycleReductionWS
    linsolve_ws::LinSolveWS
    ahat1::Array{Float64,2}
    a1copy::Array{Float64,2}
    m::Array{Float64,2}
    m00::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    m02::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    m20::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    m22::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    m1::Array{Float64,2}
    m1_1::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    m1_2::SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    m2::Array{Float64,2}
    info::Array{Int64,1}

    function CycleReductionWS(n)
        linsolve_ws = LinSolveWS(n)
        ahat1 = Array(Float64,n,n)
	a1copy = Array(Float64,n,n)
        m = Array(Float64,2*n,2*n)
        m00 = view(m,1:n,1:n)
        m02 = view(m,1:n,n+(1:n))
        m20 = view(m,n+(1:n),1:n)
        m22 = view(m,n+(1:n),n+(1:n))
        m1 = Array(Float64,n,2*n)
        m1_a0 = view(m1,1:n,1:n)
        m1_a2 = view(m1,1:n,n+(1:n))
        m2 = Array(Float64,2*n,n)
        m1_a0 = view(m2,1:n,1:n)
        m1_a2 = view(m2,1:n,n+(1:n))
        info = [0;0]
        new(linsolve_ws,ahat1,a1copy,m, m00, m02, m20, m22, m1, a0, a2, m2, info) 
    end
end

function cycle_reduction!(x::Array{Float64},a0::Array{Float64},a1::Array{Float64},a2::Array{Float64},ws::CycleReductionWS, cvg_tol::Float, max_it::Int)
    n = size(a0,1)
    n2 = n*n
    it = 0

    copy!(x,a0)
    copy!(ws.ahat1,1,B,1,length(a1))
    @inbounds ws.m1_a0 = a0
    @inbounds ws.m1_a2 = a2
    @inbounds ws.m2_a0 = a0
    @inbounds ws.m2_a2 = a2
    while true
        #        ws.m = [a0; a2]*(a1\[a0 a2])
	copy!(ws.a1copy,a1)
        linsolve_core!(ws.linsolve_ws,Ref{UInt8}('N'),ws.a1copy,ws.m1)
        gemm!('N','N',-1.0,ws.m2,ws.m1,0.0,ws.m)
        @simd for i=1:n2
            @inbounds a1[i] += ws.m02[i] + ws.m20[i]
        end
        @inbounds ws.m1_a0 += ws.m00
        @inbounds ws.m1_a2 += ws.m22
        @inbounds ws.m2_a0 = ws.m1_a0
        @inbounds ws.m2_a2 = ws.m1_a2
        @simd for i=1:n
            @inbounds ws.ahat1[i] += ws.m20[i]
        end
        crit = norm(ws.m1_a0,1)
        if crit < cvg_tol
	   # keep iterating until condition on ws.m1_a2 is met
            if norm(ws.m1_a2,1) < cvg_tol
                break
            end
        elseif isnan(crit) || it == max_it
            if crit < cvg_tol
                ws.info[1] = 4
                ws.info[2] = log(norm(ws.m1_a2,1))
            else
                ws.info[1] = 3
                ws.info[2] = log(norm(ws.m1_a1,1))
            end
            return
        end        
        it += 1
    end
    linsolve_core!(ws.linsolve_ws,Ref{UInt8}('N'),ws.ahat1,x)
    scal!(length(x),-1.0,x,1)

end

function cycle_reduction_check(x::Array{Float64,2},a0::Array{Float64,2}, a1::Array{Float64,2}, a2::Array{Float64,2},cvg_tol::Float64)
    res = a0 + a1*x + A2*x*x
    if (sum(sum(abs.(res))) > cvg_tol)
        print("the norm of the residuals, ", res, ", compared to the tolerance criterion ",cvg_tol)
    end
    nothing
end
     
end