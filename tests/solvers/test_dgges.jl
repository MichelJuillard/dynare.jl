include("dgges.jl")

using DGGES

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
