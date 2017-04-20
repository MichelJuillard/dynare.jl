using BenchmarkTools
#import Base.A_ldiv_B!
include("quasi_upper_triangular.jl")

n = 1000
a = randn(n,n)
t = lu(a)[2]
b = randn(n)

function f!(r,a,b)
    I_plus_rA_ldiv_B!(r,a,b)
end


qa = QuasiUpperTriangular(t)    
r = 2.0
f!(r,qa,b)
res = @benchmark f!(r,qa,b)
#res = @benchmark A_ldiv_B!(UpperTriangular(a),b)
display(minimum(res))

         
