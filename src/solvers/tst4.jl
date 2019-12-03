include("../../src/set_path.jl")

using LinearAlgebra
using model
using FaaDiBruno: faa_di_bruno!, partial_faa_di_bruno!, FaaDiBrunoWs
using KOrderSolver
using Test

function pane_add!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                   d_dim, s_dim, offset_d, offset_s, order)
    nc = length(i_col_s)
    if order > 1
        inc_d = d_dim^(order-1)
        inc_s = s_dim^(order-1)
        for i = 1:nc
            os = (i_col_s[i] - 1)*inc_s + offset_s
            od = (i_col_d[i] - 1)*inc_d + offset_d
            pane_add!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                      d_dim, s_dim, od, os, order-1)
        end
    else
        nr = length(i_row_d)
        for i = 1:nc
            kd = i_col_d[i] + offset_d
            ks = i_col_s[i] + offset_s
            for j = 1:nr
                dest[i_row_d[j], kd] += src[i_row_s[j], ks]
            end
        end
    end
end

function pane_add!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                   d_dim, s_dim, order)
    offset_d = 0
    offset_s = 0
    pane_add!(dest, src, i_row_d, i_row_s, i_col_d,
              i_col_s, d_dim, s_dim, offset_d, offset_s, order)
end

function pane_copy!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                    d_dim, s_dim, offset_d, offset_s, order)
    nc = length(i_col_s)
    if order > 1
        inc_d = d_dim^(order-1)
        inc_s = s_dim^(order-1)
        for i = 1:nc
            os = (i_col_s[i] - 1)*inc_s + offset_s
            od = (i_col_d[i] - 1)*inc_d + offset_d
            pane_copy!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                       d_dim, s_dim, od, os, order-1)
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
        pane_copy!(ws.gfwrd[i], g[i + k], 1:ws.nfwrd, ws.fwrd_index,
                   1:ws.nstate, 1:ws.nstate, ws.nstate, ns, 0,
                   source_offset[i], i)
    end
end

# h order: y u ϵ σ
function update_hh_current(hh, g, order, ws)
    # derivatives for g(y,u,σ)
    hh_destination_index = [collect(1:ws.nstate + ws.nshock); ws.nstate + 2*ws.nshock + 1]
    pane_copy!(hh[order], g[order], ws.nstate .+ ws.cur_index, ws.cur_index, hh_destination_index,
               1:(ws.nstate+ws.nshock+1), ws.nstate + 2*ws.nshock + 1, ws.nstate + ws.nshock + 1, order)
end

function update_hh_fwrd!(hh, g, order, y_order, destination_offset, source_offset, ws)
    ns = ws.nstate + ws.nshock
    ns1 = ws.nstate + ws.nshock + 1
    # update ws.gfwrd
    gfwrd_update!(ws, g, order, y_order, source_offset)
    # TO BE PUT IN KorderWs
    gs = [g[i][ws.state_index, :] for i = 1:y_order]
    # derivatives for g(g(y,u,σ),ϵ,σ)
    fill!(ws.gg[y_order], 0.0)
    if y_order == 1
        mul!(ws.gg[y_order], ws.gfwrd[1], gs[1])
    else
        partial_faa_di_bruno!(ws.gg[y_order], ws.gfwrd, gs, y_order, ws.faa_di_bruno_ws_1)
    end
    pane_copy!(hh[order], ws.gg[y_order], ws.nstate + ws.ncur .+ (1:ws.nfwrd), 1:ws.nfwrd,
               1:ns, 1:ns, ns1 + ws.nshock, ns1, destination_offset, 0, y_order)
end

function update_hh_fwrd_σ!(hh, g, order, y_order, destination_offset, source_offset, ws)
    ns = ws.nstate + ws.nshock
    ns1 = ws.nstate + ws.nshock + 1
    # update ws.gfwrd
    gfwrd_update!(ws, g, order, y_order, source_offset)
    gs = [g[i][ws.state_index, :] for i = 1:y_order]
    # derivatives for g(g(y,u,σ),ϵ,σ)
    fill!(ws.gg[y_order], 0.0)
    if y_order == 1
        mul!(ws.gg[y_order], ws.gfwrd[1], gs[1])
    else
        partial_faa_di_bruno!(ws.gg[y_order], ws.gfwrd, gs, y_order, ws.faa_di_bruno_ws_1)
    end
    hh_destination_index = [collect(1:ns); ns1 + ws.nshock]
    pane_add!(hh[order], ws.gg[y_order], ws.nstate + ws.ncur .+
              (1:ws.nfwrd), 1:ws.nfwrd, hh_destination_index, 1:ns1,
              ws.nstate + 2*ws.nshock + 1, ws.nstate + ws.nshock + 1,
              destination_offset, 0, y_order)
end

function update_hh_0(hh, g, order, y_order, uσ_order, index_d, index_s, stop, ws)
    if uσ_order > 1
        for i = 1:stop
            if i == ws.nshock + 1
                update_hh_1(hh, g, order, y_order, uσ_order - 1, index_d, index_s, i, ws)
            else
                update_hh_0(hh, g, order, y_order, uσ_order - 1, index_d, index_s, i, ws)
            end
            index_d[uσ_order] += 1
            for j = 1:(uσ_order-1)
                index_d[j] = ws.nstate + ws.nshock + 1
            end
            index_s[uσ_order] += 1
            for j = 1:(uσ_order-1)
                index_s[j] = nstate + 1
            end
        end
    else
        for i = 1:stop
            for j = 1:y_order
                source_offset[j] = (linear_index(index_s, ws.nng) - 1)*(ws.nstate + ws.nshock + 1)^j
            end
            destination_offset = (linear_index(index_d, ws.nnh) - 1)*(ws.nstate + 2*ws.nshock + 1)^y_order
            update_hh_fwrd!(hh, g, order, y_order, destination_offset, source_offset, ws)
            index_d[1] += 1
            index_s[1] += 1
        end
    end
end

function update_hh_1(hh, g, order, y_order, uσ_order, index_d, index_s, stop, ws)
    if uσ_order > 1
        for i = 1:stop
            update_hh_1(hh, g, order, y_order, uσ_order - 1, index_d, index_s, i, ws)
            index_d[uσ_order] += 1
            for j = 1:(uσ_order-1)
                index_d[j] = ws.nstate + ws.nshock + 1
            end
            index_s[uσ_order] += 1
            for j = 1:(uσ_order-1)
                index_s[j] = nstate + 1
            end
        end
    else
        for i = 1:stop
            for j = 1:y_order
                source_offset[j] = (linear_index(index_s, ws.nng) - 1)*(ws.nstate + ws.nshock + 1)^j
            end
            destination_offset = (linear_index(index_d, ws.nnh) - 1)*(ws.nstate + 2*ws.nshock + 1)^y_order
            update_hh_fwrd_σ!(hh, g, order, y_order, destination_offset, source_offset, ws)
            index_d[1] += 1
            index_s[1] += 1
        end
    end
end

#
# h_σσσ = g_σσσ + g_y*g_σσσ + (g_yy*kron(g_σσ,g_σ) + ...) + g_yyy*kron(g_σ,g_σ,g_σ) + g_yσ*g_σσ + g_yyσ*kron(g_σ,g_σ) + g_yσσ*g_σ
# h_sσσ = g_y*g_sσσ + (g_yy*kron(g_sσ,g_σ) + ...) + g_yyy*kron(g_s,g_σ,g_σ) + g_yσ*g_sσ + g_yyσ*kron(g_s,g_σ) + g_yσσ*g_s
# h_ssσ = g_y*g_ssσ + (g_yy*kron(g_ss,g_σ) + ...) + g_yyy*kron(g_s,g_s,g_σ) + g_yσ*g_ss + g_yyσ*kron(g_s,g_s) 
# h_su'σ = g_yu*g_sσ + g_yyu*kron(g_s,g_σ) + ...) + g_yuσ*g_s
# h_u'u'σ = g_uuσ + g_suu*g_σ
#
function update_hh_σ(hh, g, order, y_order, ws)
    for σorder = 1:(order -  y_order)
        for uorder = 1:(σorder - 1)
            for i = 1:σorder
                σorder1 = σorder - i
                y_order_1 = y_order + i
                nshock1 = ws.nshock + 1
                index_d = vcat(repeat([ws.nstate + ws.nshock + 2], uσ_order))
                index_s = vcat(repeat([ws.nstate + 1], uσ_order))
                uσ_order = u_order + σ_order
                update_hh_0(hh, g, order, y_order, uσ_order, index_d, index_s, ws.nshock + 1, ws)
            end
        end
    end
end

    
function update_hh(hh, g, order, ws)
    update_hh_current(hh, g, order, ws)
    # mark with colu;ns with NaN
#    k = ws.nstate + ws.nvar + 1
#    for i = 1:ws.nhcol^order
#        hh[order][k] = NaN
#        k += ws.nhrow
#    end
    # derivatives with respect to state variables and current shocks
    update_hh_fwrd!(hh, g, order, order, 0, repeat([0],order), ws)
    # derivatives with respect to future shocks and state variables
    # including current shocks
    for y_order = 1:(order - 1)
        uσ_order = order - y_order
        nshock1 = ws.nshock + 1
        index_d = vcat(repeat([ws.nstate + nshock1], uσ_order))
        index_s = vcat(repeat([ws.nstate + 1], uσ_order))
        update_hh_0(hh, g, order, y_order, uσ_order, index_d,
                    index_s, nshock1, ws)
    end
    # derivatives with respect only to future shocks
    pane_copy!(hh[order], ws.gg[order], ws.nstate + ws.ncur .+
               (1:ws.nfwrd), 1:ws.nfwrd, ws.ngcol .+ (1:nshock),
               ws.nstate .+ (1:nshock), ws.nstate + 2*ws.nshock + 1,
               ws.nstate + ws.nshock + 1, 0, 0, y_order)
    # derivatives with respect to σ
end

function linear_index(i, n)
    r = i[1]
    for j = 1:(length(i) - 1)
        r += (i[j+1] - 1)*n[j]
    end
    return r
end

function fill_symmetrical_derivatives(d::Matrix{Float64}, k::Int64, j::Vector{Int64}, ci, ws)
    @inbounds for i in ci
        i1 = i.I
        if !issorted(i1)
            j .= i1
            sort!(j)
            k1 = linear_index(i, ws.nnh, k)
            k2 = linear_index(j, ws.nnh, k)
            copyto!(d,(k1 - 1)*n + 1, d, (k2 - 1)*n + 1, n)
        end
    end
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
source_offset = zeros(Int64, y_order)
destination_offset = 0
for i=1:order
    fill!(hh[i], 0.0)
end
update_hh_fwrd!(hh, g, order, y_order, destination_offset, source_offset, ws)
target = zeros(nfwrd, ws.ngcol^order)
@test ws.gfwrd[1] == g[1][fwrd_index, li[1][1:nstate]]
@test ws.gfwrd[2] == g[2][fwrd_index, vec(li[2][1:nstate,1:nstate])]
@test ws.gfwrd[3] == g[3][fwrd_index, vec(li[3][1:nstate,1:nstate,1:nstate])]
gs = [g[i][ws.state_index, :] for i = 1:order]
partial_faa_di_bruno!(target, ws.gfwrd, gs, y_order, ws.faa_di_bruno_ws_1)
rows = nstate + ncur .+ (1:nfwrd)
ngcol1 = ws.nstate + ws.nshock
li1 = LinearIndices((1:ngcol, 1:ngcol, 1:ngcol))
@test hh[3][rows, vec(li2[3][1:ngcol1, 1:ngcol1, 1:ngcol1])] == target[:,vec(li1[1:ngcol1, 1:ngcol1, 1:ngcol1])]
@test all(hh[3][rows, vec(li2[3][nggcol, 1:nggcol, 1:nggcol])] .== 0)

y_order = 2
destination_offset = (ws.nstate + 2*ws.nshock)*nggcol^2
fill!(hh[3],0.0)
update_hh_fwrd_σ!(hh, g, order, y_order, destination_offset, source_offset, ws)
target = zeros(nfwrd, ws.ngcol^y_order)
partial_faa_di_bruno!(target, ws.gfwrd, gs, y_order, ws.faa_di_bruno_ws_1)
rows = nstate + ncur .+ (1:nfwrd)
ngcol1 = ws.nstate + ws.nshock
li1 = LinearIndices((1:ngcol, 1:ngcol, 1:ngcol))
@test hh[3][rows, destination_offset + 1] == target[:, 1]
k = [collect(1:(ws.nstate + ws.nshock)); ws.nstate + 2*ws.nshock + 1]
@test hh[3][rows, vec(li2[3][k, k, nggcol])] == target

for i=1:order
    fill!(hh[i], 0.0)
end

# y_order = 1 u_order = 1 σ_order = 1 u1
y_order = 1
uσ_order = 2
nshock1 = ws.nshock + 1
update_hh_1(hh, g, order, y_order, uσ_order, [7, 5],
            [5, 3], nshock1, ws)
ws.gfwrd[1] = g[3][fwrd_index, vec(li[3][1:nstate, nstate + 1, nstate + nshock + 1])]
target = ws.gfwrd[1]*gs[1][:, 1:4]
ws.gfwrd[1] = g[2][fwrd_index, vec(li[2][1:nstate, nstate + 1])]
target += ws.gfwrd[1]*gs[2][:, vec(li[2][1:4, nstate + nshock + 1])]
@test hh[3][rows, vec(li2[3][1:4, 5, 7])] == target

index_d = vcat(repeat([ws.nstate + nshock1], uσ_order))
index_s = vcat(repeat([ws.nstate + 1], uσ_order))
update_hh_0(hh, g, order, y_order, uσ_order, index_d,
            index_s, nshock1, ws)

ws.gfwrd[1] = g[3][fwrd_index, vec(li[3][1:nstate, nstate + 1, nstate + nshock + 1])]
target = ws.gfwrd[1]*gs[1][:, 1:4]
ws.gfwrd[1] = g[2][fwrd_index, vec(li[2][1:nstate, nstate + 1])]
target += ws.gfwrd[1]*gs[2][:, vec(li[2][1:4, nstate + nshock + 1])]
@test hh[3][rows, vec(li2[3][1:4, 5, 7])] == target

for i=1:order
    fill!(hh[i], 0.0)
end

update_hh(hh, g, order, ws)

# hh for current variables
rows = nstate .+ (1:nvar)
k = [1,2,3,4,7]
@test hh[3][rows, k] == g[3][:,1:5]
@test hh[3][rows, 7 .+ k] == g[3][:,6:10]
cols1 = ((nstate + 2*nshock)*(nstate + 2*nshock + 1)^2
         + (nstate + 2*nshock)*(nstate + 2*nshock + 1) .+ k)
cols2 = ((nstate + nshock)*(nstate + nshock + 1)^2 +
         (nstate + nshock)*(nstate + nshock + 1) .+ (1:5))
@test hh[3][rows, cols1] == g[3][:, cols2]

rows = nstate + ncur .+ (1:nfwrd) 
gs = [g[i][ws.state_index, :] for i = 1:order]
# y_order = 1
target = (g[3][fwrd_index, nstate*25 .+ vcat(1:2, 6:7)]*kron(g[1][state_index, :], g[1][state_index, :]))
@test hh[3][rows, 4*49 .+ vcat(1:4, 8:11)] ≈ target[:, vcat(1:4, 6:9)]

# y_order = 2 u1
target = (g[3][fwrd_index, nstate*25 .+ vcat(1:2, 6:7)]*kron(g[1][state_index, :], g[1][state_index, :]))
@test hh[3][rows, 4*49 .+ vcat(1:4, 8:11)] ≈ target[:, vcat(1:4, 6:9)]

target = zeros(nfwrd,ngcol^2)
ws.gfwrd[1] = g[2][fwrd_index, vec(li[2][1:nstate, nstate + 1])]
ws.gfwrd[2] = g[3][fwrd_index, vec(li[3][1:nstate, 1:nstate, nstate + 1])]
partial_faa_di_bruno!(target, ws.gfwrd, gs, 2, ws.faa_di_bruno_ws_1)
@test hh[3][rows, vec(li2[3][1:4, 1:4, nstate + nshock + 1])] == target[:, vec(li[2][1:4, 1:4])]

# y_order = 2 u2
ws.gfwrd[1] = g[2][fwrd_index, vec(li[2][1:nstate, nstate + 2])]
ws.gfwrd[2] = g[3][fwrd_index, vec(li[3][1:nstate, 1:nstate, nstate + 2])]
partial_faa_di_bruno!(target, ws.gfwrd, gs, 2, ws.faa_di_bruno_ws_1)
@test hh[3][rows, vec(li2[3][1:4, 1:4, nstate + nshock + 2])] == target[:, vec(li[2][1:4, 1:4])]

# y_order = 3
target = zeros(nfwrd,ngcol^3)
gfwrd_update!(ws, g, 3, 3, zeros(Int64, 3))
partial_faa_di_bruno!(target, ws.gfwrd, gs, 3, ws.faa_di_bruno_ws_1)
@test hh[3][rows, vec(li2[3][1:4, 1:4, 1:4])] == target[:, vec(li[3][1:4, 1:4, 1:4])]

# y_order = 1 u_order = 1 σ_order = 1 u1
wws.gfwrd[1] = g[3][fwrd_index, vec(li[3][1:nstate, nstate + 1, nstate + nshock + 1])]
target = ws.gfwrd[1]*gs[1][:, 1:4]
ws.gfwrd[1] = g[2][fwrd_index, vec(li[2][1:nstate, nstate + 1])]
target += ws.gfwrd[1]*gs[2][:, vec(li[2][1:4, nstate + nshock + 1])]
@test hh[3][rows, vec(li2[3][1:4, 5, 7])] == target

# y_order = 1 u_order = 1 σ_order = 1 u2
ws.gfwrd[1] = g[3][fwrd_index, vec(li[3][1:nstate, nstate + 2, nstate + nshock + 1])]
target = ws.gfwrd[1]*gs[1][:, 1:4]
ws.gfwrd[1] = g[2][fwrd_index, vec(li[2][1:nstate, nstate + 2])]
target += ws.gfwrd[1]*gs[2][:, vec(li[2][1:4, nstate + nshock + 1])]
@test hh[3][rows, vec(li2[3][1:4, 6, 7])] == target

# y_order = 1 σ_order = 2
ws.gfwrd[1] = g[3][fwrd_index, vec(li[3][1:nstate, nstate + nshock + 1, nstate + nshock + 1])]
target = ws.gfwrd[1]*gs[1][:, 1:4]
ws.gfwrd[1] = g[2][fwrd_index, vec(li[2][1:nstate, nstate + nshock + 1])]
target += ws.gfwrd[1]*gs[2][:, vec(li[2][1:4, nstate + nshock + 1])]

@test hh[3][rows, vec(li2[3][1:4, 5, 7])] == target

