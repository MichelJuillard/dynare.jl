include("dgges.jl")

using DGGES

A = [-2. 1. 3.; 2. 1. -1.; -7. 2. 7.]

ws = DgeesWS(A)

Aold = copy(A)
dgees!(ws, A)
println(ws.vs)
println(norm(Aold - ws.vs*A*ws.vs'))
assert(A == [2.0 0.801792 6.63509; -8.55988e-11 2.0 8.08286; 0.0 0.0 1.99999])
assert(ws.vs == [0.577351 0.154299 -0.801784; 0.577346 -0.77152 0.267262; 0.577354 0.617211 0.534522])
assert(ws.eigenvalues == Complex{Float64}[2.0+8.28447e-6im,2.0-8.28447e-6im,1.99999+0.0im])

n = 3
A = eye(n)
A[3,3] = 0.5
A[2,2] = 0.5
B = eye(n)
B[1,1] = 0.5

ws = DggesWS(A,B)

sdim = 0

vsr = Array(Float64,n,n)
vsl = Array(Float64,n,n)
eigval = Array(Complex64,n)

dgges!('N', 'V', A, B, vsl, vsr,eigval,ws)

println(A)
println(B)
println(vsr)
println(eigval)
println(ws.sdim)
println(ws.info)

