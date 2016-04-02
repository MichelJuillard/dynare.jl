# a model with two equations
function f(x,ieq,n)
    y = zeros(Float64,n)
    i = 1
    if ieq & 1 != 0
        y[i] = x[1]^2
        i += 1
    end
    if ieq & 2 != 0
        y[i] = x[1] - x[2]
        i += 1
    end
    return y
end

# making a model with n equations
# all n equations are identical but it doesn't matter here
function make_fff(n)
    f = """function fff(x,ieq,n)
  y = zeros(Float64,n)
  i = 1
"""
    for i=1:n
        j = Int(floor(i/128) + 1)
        k = i % 128
        f *= """
  if ieq[$j] & (UInt128(1) << $k) != 0
    y[i] = x[1]^2
  end
"""
     end
     f *= "  return y\nend"
     # parsing the string into a Julia expression
     return parse(f)
end

# entering an individual equation
function eq1(x)
    x[1]^2
end

# entering an another individual equation
function eq2(x)
    x[1] - x[2]
end

# stacking n pointers to the first equation
function make_eqs(n)
    eqs = Function[]
    for i=1:n
        push!(eqs,eq1)
    end
    return eqs
end

# returning equation result from the pointers 
function g(x,eqs,ieq,n)
    y = zeros(Float64,n)
    for i = 1:n
        y[i] = eqs[ieq[i]](x)
    end
    return y
end

# replicating n times the evaluation of the model
# with the if approach
function t1(n,x,ieq,m)
    tic()
    for i = 1:n
        z = ff(x,ieq,m)
    end
    toc()
end

# replicating n times the evaluation of the model
# with the pointer approach
function t2(n,x,eqs,ieq,m)
    tic()
    for i = 1:n
        z = g(x,eqs,ieq,m)
    end
    toc()
end

# making a 2 equaton model with pointers by hand
eqs = Function[]
push!(eqs,eq1)
push!(eqs,eq2)

# checking that the results are the same
ff = f
x = [1 10]
ieq = 1
println(ff(x,ieq,1))
println(g(x,eqs,ieq,1))
ieq = 2
println(ff(x,ieq,1))
println(g(x,eqs,ieq,1))
ieq = 3
println(ff(x,ieq,2))
ieq = [1, 2]
println(g(x,eqs,ieq,2))

    
# making models with 3000 equations

fff = eval(make_fff(3000))
ff = fff
eqs = make_eqs(3000)

# evaluating only 2 equations
ieq = zeros(UInt128,Int(ceil(3000/128)))
ieq[1] = 3
t1(1,x,ieq,2)
t1(50000,x,ieq,2)
ieq = [1, 2]
t2(50000,x,eqs,ieq,2)

# evalutating 3000 equations
ieq = ones(UInt128,Int(ceil(3000/128)))
fff(x,ieq,2)
t1(50000,x,ieq,3000)            
ieq = 1:3000
t2(50000,x,eqs,ieq,3000)

