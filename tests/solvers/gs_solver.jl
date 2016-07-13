include("dgges.jl")
using DGGES


function gs_solver(D,E,qz_criterium,k1,k2,check=false)

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
