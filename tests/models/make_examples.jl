using JLD

DYNARE_ROOT = "/data/projects/dynare/git/preprocessor/src"
cd("example1")
run(`$DYNARE_ROOT/dynare_m example1.mod language=julia output=first`)

include("example1/example1.jl")

model_ = example1.model_

endo_steady_state = Vector{Float64}(undef, model_.endo_nbr)
exo_steady_state = zeros(3, model_.exo_nbr)
model_.steady_state(endo_steady_state, exo_steady_state[1, :], model_.params)
println(endo_steady_state)

T = Vector{Float64}(undef, sum(model_.temporaries.dynamic[1:2]))
residual = Vector{Float64}(undef, model_.endo_nbr)
nn = count(model_.lead_lag_incidence .!= 0) 
g1 = Matrix{Float64}(undef, model_.endo_nbr, nn + model_.exo_nbr)
y = Vector{Float64}(undef, nn)
k = findall(model_.lead_lag_incidence[1,:] .> 0)
m1 = length(k)
y[1:m1] = endo_steady_state[k]
k = findall(model_.lead_lag_incidence[2,:] .> 0)
m2 = length(k)
y[m1 .+ (1:m2)] = endo_steady_state[k]
k = findall(model_.lead_lag_incidence[3,:] .> 0)
m3 = length(k)
y[m1 + m2 .+ (1:m3)] = endo_steady_state[k]


model_.dynamic(T, residual, g1, y, exo_steady_state,
               model_.params, endo_steady_state, 2)

save("jacobian.jld", "jacobian", g1)

cd("..")
