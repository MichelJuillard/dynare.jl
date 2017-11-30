    using Base.Test

include("linsolve_algo.jl")

n = 10

ws = linsolve_algo.LinSolveWS(n)
a = randn(n,n)
a_orig = copy(a)
b = randn(n)
b_orig = copy(b)
linsolve_algo.linsolve_core!(ws,Ref{UInt8}('N'),a,b)
@test b ≈ a_orig\b_orig

a = randn(n,n)
a_orig = copy(a)
b = randn(n)
b_orig = copy(b)
c = randn(n)
c_orig = copy(c)
linsolve_algo.linsolve_core!(ws,Ref{UInt8}('N'),a,b,c)
@test b ≈ a_orig\b_orig
@test c ≈ a_orig\c_orig

a = randn(n,n)
a_orig = copy(a)
b = randn(n)
b_orig = copy(b)
c = randn(c)
c_orig = copy(c)
d = randn(n)
d_orig = copy(d)
linsolve_algo.linsolve_core!(ws,Ref{UInt8}('N'),a,b,c,d)
@test b ≈ a_orig\b_orig
@test c ≈ a_orig\c_orig
@test d ≈ a_orig\d_orig

