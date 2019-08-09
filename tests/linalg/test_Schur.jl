push!(LOAD_PATH,"../src/linalg/")
using LinearAlgebra
using SchurAlgo
using Test

A = [-2. 1. 3.; 2. 1. -1.; -7. 2. 7.]

ws = DgeesWS(A)


Aold = copy(A)
dgees!(ws, A)
@test Aold ≈ ws.vs*A*ws.vs'

n = 3
A = Matrix{Float64}(I, n, n)
A[3,3] = 0.5
A[2,2] = 0.5
B = Matrix{Float64}(I, n, n)
B[1,1] = 0.5

ws = DggesWS(A,B)

sdim = 0

vsr = Array{Float64}(undef, n, n)
vsl = Array{Float64}(undef, n, n)
eigval = Array{ComplexF64}(undef, n)
dgges!('N', 'V', A, B, vsl, vsr, eigval, ws)

println(A)
println(B)
println(vsr)
println(eigval)
println(ws.sdim)


