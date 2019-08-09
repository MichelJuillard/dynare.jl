push!(LOAD_PATH,"../src/taylor")
push!(LOAD_PATH,"../src/linalg")
using ForwardDiff
using Test
using TensorOperations
using KroneckerUtils

function f(x)
    [exp(x[1])*exp(2*x[2]), 2*exp(x[1])*exp(x[2])]
end

function g(x)
    [x[1]^4*x[2]^3, x[1]^2*x[2]]
end

fg(x) = f(g(x))


df1(x) = ForwardDiff.jacobian(f,x)
df2(x) = ForwardDiff.jacobian(df1,x)
df3(x) = ForwardDiff.jacobian(df2,x)
df4(x) = ForwardDiff.jacobian(df3,x)
dg1(x) = ForwardDiff.jacobian(g,x)
dg2(x) = ForwardDiff.jacobian(dg1,x)
dg3(x) = ForwardDiff.jacobian(dg2,x)
dg4(x) = ForwardDiff.jacobian(dg3,x)
dfg1(x) = ForwardDiff.jacobian(fg,x)
dfg2(x) = ForwardDiff.jacobian(dfg1,x)
dfg3(x) = ForwardDiff.jacobian(dfg2,x)
dfg4(x) = ForwardDiff.jacobian(dfg3,x)

using FaaDiBruno
faadibruno_ws = FaaDiBrunoWs(2,2,3)
println(faadibruno_ws.recipees[3][2])
faadibruno_ws = FaaDiBrunoWs(2,2,4)
println(faadibruno_ws.recipees[3][2])


n = 2
dfg=zeros(n, n^3)
x = randn(n)
ff=[ df1(g(x)), reshape(df2(g(x)),n,n*n), reshape(df3(g(x)),n,n^3), reshape(df4(g(x)),n,n^4)]
gg=[ dg1(x), reshape(dg2(x),n,n*n), reshape(dg3(x),n,n^3), reshape(dg4(x),n,n^4)]


FaaDiBruno.apply_recipees!(dfg,faadibruno_ws.recipees[3][2],ff[2],gg,3,faadibruno_ws)
t1 = ff[2]*kron(gg[2],gg[1])
t2 = t1 + t1[:,[1, 3, 2, 4, 5, 7, 6, 8]] + t1[:,[1, 3, 5, 7, 2, 4, 6, 8]]
@test dfg ≈ t2
t3 = zeros(2,2,2,2)
ff1 = reshape(ff[1],2,2)
ff2 = reshape(ff[2],2,2,2)
ff3 = reshape(ff[3],2,2,2,2)
ff4 = reshape(ff[4],2,2,2,2,2)
gg1 = reshape(gg[1],2,2)
gg2 = reshape(gg[2],2,2,2)
gg3 = reshape(gg[3],2,2,2,2)
gg4 = reshape(gg[4],2,2,2,2,2)

@tensor t3[a,b,c,d] = ff2[a,e,f]*gg2[e,b,c]*gg1[f,d] + ff2[a,e,f]*gg2[e,b,d]*gg1[f,c] + ff2[a,e,f]*gg2[e,d,c]*gg1[f,b]

@test dfg ≈ reshape(t3,2,8)

t10 = zeros(2,2,2,2,2)
t20 = zeros(2,2,2,2,2)
t30 = zeros(2,2,2,2,2)
t30a = zeros(2,2,2,2,2)
t30b = zeros(2,2,2,2,2)
t30c = zeros(2,2,2,2,2)
t30d = zeros(2,2,2,2,2)
t30e = zeros(2,2,2,2,2)
t30f = zeros(2,2,2,2,2)
t40 = zeros(2,2,2,2,2)

@tensor begin
    t10[a,b,c,d,e] = ff1[a,f]*gg4[f,b,c,d,e]
    t20[a,b,c,d,e] = ff2[a,f,g]*gg2[f,b,c]*gg2[g,d,e] + ff2[a,f,g]*gg2[f,b,d]*gg2[g,c,e] + ff2[a,f,g]*gg2[f,b,e]*gg2[g,d,c] +
        ff2[a,f,g]*gg3[f,b,c,d]*gg1[g,e] +  ff2[a,f,g]*gg3[f,b,c,e]*gg1[g,d] +  ff2[a,f,g]*gg3[f,b,e,d]*gg1[g,c] +  ff2[a,f,g]*gg3[f,e,c,d]*gg1[g,b]
    t30[a,b,c,d,e] = ff3[a,f,g,h]*gg2[f,b,c]*gg1[g,d]*gg1[h,e] + ff3[a,f,g,h]*gg2[f,b,d]*gg1[g,c]*gg1[h,e] + ff3[a,f,g,h]*gg2[f,b,e]*gg1[g,d]*gg1[h,c] +
        ff3[a,f,g,h]*gg2[f,c,d]*gg1[g,b]*gg1[h,e] +ff3[a,f,g,h]*gg2[f,c,e]*gg1[g,d]*gg1[h,b] +ff3[a,f,g,h]*gg2[f,d,e]*gg1[g,b]*gg1[h,c]
    t30a[a,b,c,d,e] = ff3[a,f,g,h]*gg2[f,b,c]*gg1[g,d]*gg1[h,e]
    t30b[a,b,c,d,e] = ff3[a,f,g,h]*gg2[f,b,d]*gg1[g,c]*gg1[h,e]
    t30c[a,b,c,d,e] = ff3[a,f,g,h]*gg2[f,b,e]*gg1[g,d]*gg1[h,c]
    t30d[a,b,c,d,e] = ff3[a,f,g,h]*gg2[f,c,d]*gg1[g,b]*gg1[h,e]
    t30e[a,b,c,d,e] = ff3[a,f,g,h]*gg2[f,c,e]*gg1[g,d]*gg1[h,b]
    t30f[a,b,c,d,e] = ff3[a,f,g,h]*gg2[f,d,e]*gg1[g,b]*gg1[h,c]
    t40[a,b,c,d,e] = ff4[a,f,g,h,j]*gg1[f,b]*gg1[g,c]*gg1[h,d]*gg1[j,e]
end
t50 = t10 + t20 + t30 + t40
t50 = reshape(t50,2,16)
@test reshape(dfg4(x),2,16) ≈ t50

work1 = zeros(n,n^4)
work2 = zeros(n^5)
a_mul_kron_b!(work1,ff[3],gg[[2,1,1]],work2)
println("a_mul_kron_b!")
println(work1[1])
println("tensor product")
println(t30a[1])

dfg = zeros(n,n^4)
FaaDiBruno.apply_recipees!(dfg,faadibruno_ws.recipees[4][3],ff[3],gg,4,faadibruno_ws)
@test dfg ≈ reshape(t30,2,16)

dfg = zeros(n,n^4)
FaaDiBruno.apply_recipees!(dfg,faadibruno_ws.recipees[4][2],ff[2],gg,4,faadibruno_ws)
@test dfg ≈ reshape(t20,2,16)

dfg=zeros(n,n^3)
FaaDiBruno.faa_di_bruno!(dfg,ff,gg,3,faadibruno_ws)
target = reshape(dfg3(x),2,8)
@test dfg ≈ target

println("OK1")
println(t10[1])
println(t10[1]+t20[1])
println(t10[1]+t20[1]+t30[1])
println(t10[1]+t20[1]+t30[1]+t40[1])
println("OK2")
println([t30a[1], t30b[1], t30c[1], t30d[1], t30e[1], t30f[1]])
dfg=zeros(n,n^4)
FaaDiBruno.faa_di_bruno!(dfg,ff,gg,4,faadibruno_ws)
target = reshape(dfg4(x),2,16)
@test dfg ≈ target
