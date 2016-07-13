using MAT
include("gs_solver.jl")
include("gs_solver1.jl")

file = matopen("jacobian.mat")
jacobian = read(file,"jacobia")

n = 6

file = matopen("DE.mat")
D = read(file,"D")
E = read(file,"E")
D1 = copy(D)
E1 = copy(E)

@time gx =  gs_solver(E,D,1+1e-6,1:3,1:3)
@time gx =  gs_solver(E,D,1+1e-6,1:3,1:3)
@time gx1 =  gs_solver1(E1,D1,1+1e-6,1:3,1:3)
@time gx1 =  gs_solver1(E1,D1,1+1e-6,1:3,1:3)


inv_order = [5, 6, 2, 3, 1, 4]
res = D*[gx[1:3,:]; gx[4:6,:]*gx[1:3,:]] - E*[eye(3);gx[4:6,:]]
println(sum(sum(abs(res))))
res1 = D*[gx[1:3,:]; gx[4:6,:]*gx[1:3,:]] - E*[eye(3);gx[4:6,:]]
println(sum(sum(abs(res1))))

