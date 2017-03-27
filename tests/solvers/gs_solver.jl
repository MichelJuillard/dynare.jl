include("dgges.jl")
using DGGES

type GsSolverWS:
    dgges_ws::DggesWS
    vsr::Array{Float64,2}
    eigen::Array{Complex64,1}
    function GsSolverWS(a,b)
        n = size(a,1)
        ws = DggesWs(a,b)
        vsr = Array(Float64,n,n)
        eigen = Array(Complex64,n)
    end
end

function gs_solver!(d::Array{Float64,2},e::Array{Float64,2},qz_criterium::Float64,k1::Array{Int64,1},k2::Array{Int64,1},check::Bool=false,ws::GsSolverWS)

    dgges!('N','V','S',d,e,ws.vsl,ws.vsr,ws.eigen,ws.dgges_ws)
    GS = schurfact(D,E)
    s = abs(GS[:alpha]./GS[:beta]) .< qz_criterium
    # number of stable roots
    ns = sum(s)
    println("number stable roots: ",ns)
    println(abs(GS[:alpha]./GS[:beta]))
    ordschur!(GS,s)
    
    gx = -(GS[:Z][1:ns,ns+1:end]/GS[:Z][ns+1:end,ns+1:end])'
    hx1 = GS[:Z][1:ns,1:ns]/GS[:T][1:ns, 1:ns]
    hx2 = GS[:S][1:ns,1:ns]/GS[:Z][1:ns,1:ns]
    hx = hx1*hx2
    gx = [hx[k1,:]; gx[k2,:]]
end
