include("../../src/set_path.jl")

using LinearAlgebra
using model
using FaaDiBruno: faa_di_bruno!, partial_faa_di_bruno!, FaaDiBrunoWs
using KOrderSolver
using Test

function pane_copy!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                      d_dim, s_dim, offset_d, offset_s, order)
    nc = length(i_col_s)
    if order > 1
        os = offset_s
        od = offset_d
        inc_d = d_dim^(order-1)
        inc_s = s_dim^(order-1)
        for i = 1:nc
            pane_copy!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                       d_dim, s_dim, od, os, order-1)
            od += inc_d
            os += inc_s
        end
    else
        nr = length(i_row_d)
        for i = 1:nc
            kd = i_col_d[i] + offset_d
            ks = i_col_s[i] + offset_s
            for j = 1:nr
                dest[i_row_d[j], kd] = src[i_row_s[j], ks]
            end
        end
    end
end

function pane_copy!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                    d_dim, s_dim, order)
    offset_d = 0
    offset_s = 0
    pane_copy!(dest, src, i_row_d, i_row_s, i_col_d,
               i_col_s, d_dim, s_dim, offset_d, offset_s, order)
end    

function gfwrd_update!(ws, g, order, y_order, source_offset)
#    @assert y_order > 1
    k = order - y_order
    ns = ws.nstate + ws.nshock + 1
    for i = 1:y_order
        #        KOrderSolver.pane_copy!(ws.gfwrd[i], g[i + k], 1:ws.nfwrd, ws.fwrd_index, 1:ws.nstate, 1:ws.nstate, ws.nstate, ns, 0, source_offset[i], i)
        pane_copy!(ws.gfwrd[i], g[i + k], 1:ws.nfwrd, ws.fwrd_index, 1:ws.nstate, 1:ws.nstate, ws.nstate, ns, 0, source_offset[i], i)
    end
end

function update_hh_current(hh, g, order, ws)
    # derivatives for g(y,u,σ)
#    KOrderSolver.pane_copy!(hh[order], g[order], ws.nstate .+ ws.cur_index, ws.cur_index, 1:(ws.nstate+ws.nshock),
    pane_copy!(hh[order], g[order], ws.nstate .+ ws.cur_index, ws.cur_index, 1:(ws.nstate+ws.nshock+1),
                            1:(ws.nstate+ws.nshock+1), ws.nstate + 2*ws.nshock + 1, ws.nstate + ws.nshock + 1, order)
end

function update_hh_fwrd!(hh, g, order, y_order, destination_offset, source_offset, ws)
    ns = ws.nstate + ws.nshock + 1
    # update ws.gfwrd
    gfwrd_update!(ws, g, order, y_order, source_offset)
    gs = [g[i][ws.state_index, :] for i = 1:y_order]
    fill!(ws.gg[y_order], 0.0)
    if y_order == 1
        mul!(ws.gg[y_order], ws.gfwrd[1], gs[1])
    else
        partial_faa_di_bruno!(ws.gg[y_order], ws.gfwrd, gs, y_order, ws.faa_di_bruno_ws_1)
    end
    # derivatives for g(g(y,u,σ),ϵ,σ)
    pane_copy!(hh[order], ws.gg[y_order], ws.nstate + ws.ncur .+ (1:ws.nfwrd), 1:ws.nfwrd, 1:ns,
               1:ns, ws.nstate + 2*ws.nshock + 1, ws.nstate + ws.nshock + 1,
               destination_offset, 0, y_order)
end

function update_hh_0(hh, g, order, y_order, uσ_order, index_d, li_d, index_s, li_s, ws)
    if uσ_order > 1
        for i = 1:ws.nshock
            update_hh_0(hh, g, order, y_order, uσ_order - 1, index_d, li_d, index_s, li_s, ws)
            index_d[uσ_order] += 1
            for j = 1:(uσ_order-1)
                index_d[j] = ws.nstate + ws.nshock + 2
            end
            index_s[uσ_order] += 1
            for j = 1:(uσ_order-1)
                index_s[j] = nstate + 1
            end
        end
    else
        for i = 1:ws.nshock
            for j = 1:y_order
                source_offset[j] = (li_s[index_s...] - 1)*(ws.nstate + ws.nshock + 1)^j
            end
            destination_offset = (li_d[index_d...] - 1)*(ws.nstate + 2*ws.nshock + 1)^y_order
            update_hh_fwrd!(hh, ws.gg, g, order, y_order, destination_offset, source_offset, ws)
            index_d[1] += 1
            index_s[1] += 1
        end
    end
end
    
function update_hh(hh, g, order, ws)
    update_hh_current(hh, g, order, ws)
    for y_order = 1:(order - 1)
        uσ_order = order - y_order
        nshock1 = ws.nshock + 1
        li_d = LinearIndices(tuple(repeat([1:(ws.nstate + 2*ws.nshock + 1)], uσ_order)...))
        li_s = LinearIndices(tuple(repeat([1:(ws.nstate + ws.nshock + 1)], uσ_order)...))
        index_d = vcat(repeat([ws.nstate + ws.nshock + 2], uσ_order))
        index_s = vcat(repeat([ws.nstate + 1], uσ_order))
        update_hh_0(hh, g, order, y_order, uσ_order, index_d, li_d, index_s, li_s, ws)
    end
    update_hh_fwrd!(hh, g, order, order, 0, repeat([0],order), ws)
end

nvar = 5
ncur = 5
nfwrd = 2
nstate = 2
nshock = 2
fwrd_index = [1, 2]
state_index = [3, 5]
cur_index = collect(1:5)
state_range = 1:2
order = 3
ws =     KOrderWs(nvar,nfwrd,nstate,ncur,nshock,fwrd_index,state_index,cur_index,state_range,order)

ngcol = nstate + nshock + 1 
nggcol = ngcol + nstate

g = [randn(nvar, ngcol^i) for i=1:order]
hh =[Matrix{Float64}(undef, nstate + nvar + nfwrd + nshock, nggcol^i) for i=1:order]

li = [LinearIndices(tuple(repeat([1:ngcol], i)...)) for i=1:order]
li2 = [LinearIndices(tuple(repeat([1:nggcol], i)...)) for i=1:order]

y_order = 3
source_offset = zeros(Int64,order)
gfwrd_update!(ws, g, order, y_order, source_offset)
@test ws.gfwrd[1] == g[1][fwrd_index, li[1][1:nstate]]
@test ws.gfwrd[2] == g[2][fwrd_index, vec(li[2][1:nstate,1:nstate])]
@test ws.gfwrd[3] == g[3][fwrd_index, vec(li[3][1:nstate,1:nstate,1:nstate])]

y_order = 2
source_offset[1] = li[2][ngcol, nstate]
source_offset[2] = li[3][ngcol, 1, nstate + 1]
gfwrd_update!(ws, g, order, y_order, source_offset)
@test ws.gfwrd[1] == g[2][fwrd_index, source_offset[1] .+ li[1][1:nstate]]
@test ws.gfwrd[2] == g[3][fwrd_index, source_offset[2] .+ vec(li[2][1:nstate,1:nstate])]


y_order = 3
source_offset[1] = 0
source_offset[2] = 0
destination_offset = 0
for i=1:order
    fill!(hh[i], 0.0)
end
@time update_hh_fwrd!(hh, g, order, y_order, destination_offset, source_offset, ws)

for i=1:order
    fill!(hh[i], 0.0)
end

update_hh(hh, g, order, ws)

# hh for current variables
@test hh[3][nstate .+ (1:nvar), 1:5] == g[3][:,1:5]
@test hh[3][nstate .+ (1:nvar), 8:12] == g[3][:,6:10]
rows = nstate .+ (1:nvar)
cols1 = ((nstate + nshock)*(nstate + 2*nshock + 1)^2
         + (nstate + nshock)*(nstate + 2*nshock + 1) .+ (1:5))
cols2 = ((nstate + nshock)*(nstate + nshock + 1)^2 +
         (nstate + nshock)*(nstate + nshock + 1) .+ (1:5))
@test hh[3][rows, cols1] == g[3][:, cols2]

rows = nstate + ncur .+ (1:nfwrd) 
gs = [g[i][ws.state_index, :] for i = 1:order]
# y_order = 1
target = (g[3][fwrd_index, nstate*25 .+ vcat(1:2, 6:7)]*kron(g[1][state_index, :], g[1][state_index, :]))
@test hh[3][rows, 5*49 .+ vcat(1:5, 8:12)] ≈ target[:, 1:10]

# y_order = 2 u1
target = (g[3][fwrd_index, nstate*25 .+ vcat(1:2, 6:7)]*kron(g[1][state_index, :], g[1][state_index, :]))
@test hh[3][rows, 5*49 .+ vcat(1:5, 8:12)] ≈ target[:, 1:10]

target = zeros(nfwrd,ngcol^2)
ws.gfwrd[1] = g[2][fwrd_index, vec(li[2][1:nstate, nstate + 1])]
ws.gfwrd[2] = g[3][fwrd_index, vec(li[3][1:nstate, 1:nstate, nstate + 1])]
partial_faa_di_bruno!(target, ws.gfwrd, gs, 2, ws.faa_di_bruno_ws_1)
@test hh[3][rows, vec(li2[3][1:ngcol, 1:ngcol, nstate + nshock + 2])] == target

# y_order = 2 u2
ws.gfwrd[1] = g[2][fwrd_index, vec(li[2][1:nstate, nstate + 2])]
ws.gfwrd[2] = g[3][fwrd_index, vec(li[3][1:nstate, 1:nstate, nstate + 2])]
partial_faa_di_bruno!(target, ws.gfwrd, gs, 2, ws.faa_di_bruno_ws_1)
@test hh[3][rows, vec(li2[3][1:ngcol, 1:ngcol, nstate + nshock + 3])] == target

# y_order = 3
target = zeros(nfwrd,ngcol^3)
gfwrd_update!(ws, g, 3, 3, zeros(Int64, 3))
partial_faa_di_bruno!(target, ws.gfwrd, gs, 3, ws.faa_di_bruno_ws_1)
@test hh[3][rows, vec(li2[3][1:ngcol, 1:ngcol, 1:ngcol])] == target


