module Tmp

using Dynare

function make_gsσi_1!(gsσi::Matrix{Float64}, g::Matrix{Float64}, nstate, nshock, state_index, k, i, rs, rd, inc)
    if k > 1
        rs_ = rs
        rd_ = rd
        for j = 1:(nstate + nshock)
            make_gsσi_1!(gsσi, g, nstate, nshock, state_index, k - 1, i, rs_, rd_, inc)
            rs_ += inc^(k + i -1)
            rd_ += (nstate + nshock)^(k - 1)
        end
    else
        v1 = view(g, state_index, rs)
        v2 = view(gsσi, :, rd)
        v1 .= v2
    end
end

function make_gsσi!(gsσi::Matrix{Float64}, g::Matrix{Float64}, nstate, nshock, state_index, k, i)
    inc = nstate + nshock + 1
    rs = inc^i:inc^i:((nstate + nshock)*inc^i)
    rd = 1:(nstate + nshock)
    make_gsσi_1!(gsσi, g, nstate, nshock, state_index, k, i, rs, rd, inc)
end

function make_gsσ!(gsσ::Vector{Matrix{Float64}}, g::Vector{Matrix{Float64}}, nstate, nshock, state_index, k, j)
    for i = 1:k
        make_gsσi!(gsσ[i], g[i], nstate, nshock, state_index, k, j)
    end
end

function make_dkj!(dkj::Matrix{Float64}, g::Vector{Matrix{Float64}}, k::Int64, j::Int64, gfykσlΣ::Vector{Matrix{Float64}},
                  Sigma::Vector{Vector{Float64}}, nstate::Int64,
                  nfwrd::Int64, nshock::Int64, fwrd_index::Vector{Int64}, state_index::Vector{Int64},
                  gykfulσm::Matrix{Float64}, faa_di_bruno_ws::Dynare.FaaDiBruno.FaaDiBrunoWs, dkji::Matrix{Float64}, gsσ::Vector{Matrix{Float64}})
    for m = 2:j
        gul = view(gykfulσm,:,1:nshock^m)
        for i = 0:j-m
            if k > 0
                for q = 1:k
                    Dynare.Solvers.KOrderSolver.make_gfykσlΣm!(gfykσlΣ[q], g[k+j-i], Sigma[m], nstate,
                                                               nfwrd, nshock, fwrd_index, q, m, j - m - i, gul)
                end
                make_gsσ!(gsσ, g, nstate, nshock, state_index, k, i)
                Dynare.FaaDiBruno.faa_di_bruno!(dkji, gfykσlΣ, gsσ, k, faa_di_bruno_ws)
            else
                Dynare.Solvers.KOrderSolver.make_gfykσlΣm!(dkji, g[j-i], Sigma[m], nstate,
                                                           nfwrd, nshock, fwrd_index, 0, m, j - m - i, gul)
            end                
            dkj .+= dkji
        end
    end
end

function make_gsσ(hh, gg, g, order, ws)
    ws
    for i=1:order
        make_gg!(gg,g,order,ws)
        make_hh!(hh, g, gg, order, ws)
    end
end
        
    
