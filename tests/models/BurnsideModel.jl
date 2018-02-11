module BurnsideModel
import ForwardDiff
 
export beta, rho, theta, xbar, x_ss, y_ss, sd_e, sigma2, df, burnside_exact_solution, dgx, dgx2, dgx3, dgx4, dgs2, dgs4 

# Burnside model

# variables order: x y

beta = 0.95;
rho = -0.139;
theta = -1.5;
xbar = 0.0179;

function f_orig(x_m1,x,y,x_p1,y_p1,e)
    e1 = y - beta*exp(theta*x_p1)*(1+y_p1)
    e2 = x - (1-rho)*xbar + rho*x_m1 + e
    [e1, e2]
end

function f(x)
    e1 = x[3] - beta*exp(theta*x[4])*(1+x[5])
    e2 = x[2] - (1-rho)*xbar + rho*x[1] + x[6]
    [e1, e2]
end

# steady_state
x_ss = xbar
y_ss = beta*exp(theta*xbar)/(1-beta*exp(theta*xbar))

# standard deviation of shocks
sd_e = 0.0348
sigma2 = sd_e*sd_e
sigma4 = 3*sd_e^4

# derivatives
df1(x) = ForwardDiff.jacobian(f,x)
df2(x) = ForwardDiff.jacobian(df1,x)
#df3(x) = ForwardDiff.jacobian(df2,x)
#df4(x) = ForwardDiff.jacobian(df3,x)

z = [x_ss, x_ss, y_ss, x_ss, y_ss, 0]
tmp = reshape([df2(z)[i + j + k*12] for i=1:2:12 for k=0:5 for j=0:1],2,36)
df = [df1(z), tmp]

# exact solution
function burnside_exact_solution(x,beta,theta,rho,xbar,sigma2)
    
    n = 800

    if beta*exp(theta*xbar+.5*theta^2*sigma2/(1-rho)^2) > 1-eps()
        println("The model doesn't have a solution!")
        return
    end
    
    i = collect(1:n)
    a = theta*xbar*i+(theta^2*sigma2)/(2*(1-rho)^2)*(i-2*rho*(1-rho.^i)/(1-rho)+rho^2*(1-rho.^(2*i))/(1-rho^2))
    b = theta*rho*(1-rho.^i)/(1-rho)
    
    xhat = x-xbar
    
    n2 = length(x)
    
    y = sum(beta.^i.*exp.(a+b*xhat))
end

function g(x,sigma2)
    x1 = (1-rho)*xbar + rho*x[1] + x[2]
    y = burnside_exact_solution(x1,beta,theta,rho,xbar,sigma2)
end

dgx = (theta/(1-rho))*(beta*rho*exp(theta*xbar)*(1/(1-beta*exp(theta*xbar)) - rho/(1-beta*rho*exp(theta*xbar))))
dgx2 = (theta/(1-rho))^2*beta*rho^2*exp(theta*xbar)*(1/(1-beta*exp(theta*xbar)) - 2*rho/(1-beta*rho*exp(theta*xbar)) + rho^2/(1-beta*rho^2*exp(theta*xbar)))
dgx3 = (theta/(1-rho))^3*beta*rho^3*exp(theta*xbar)*(1/(1-beta*exp(theta*xbar)) - 3*rho/(1-beta*rho*exp(theta*xbar)) + 3*rho^2/(1-beta*rho^2*exp(theta*xbar)) - rho^3/(1-beta*rho^3*exp(theta*xbar)))
dgx4 = (theta/(1-rho))^4*beta*rho^4*exp(theta*xbar)*(1/(1-beta*exp(theta*xbar)) - 4*rho/(1-beta*rho*exp(theta*xbar)) + 6*rho^2/(1-beta*rho^2*exp(theta*xbar))
                                                     - 4*rho^3/(1-beta*rho^3*exp(theta*xbar)) + rho^4/(1-beta*rho^4*exp(theta*xbar)))

dgs2 = ((theta^2*beta*exp(theta*xbar))/(1-rho)^2)*(1/(1-beta*exp(theta*xbar))^2 - (2/(1-rho))*(rho/(1-beta*exp(theta*xbar))-rho^2/(1-beta*rho*exp(theta*xbar))) + (1/(1-rho^2))*(rho^2/(1-beta*exp(theta*xbar))-rho^4/(1-beta*rho^2*exp(theta*xbar))))*sigma2

dgs4 = ((theta^4*beta*exp(theta*xbar))/(1-rho)^4)*(1/(1-beta*exp(theta*xbar))^2 - (4/(1-rho))*(rho/(1-beta*exp(theta*xbar))-rho^2/(1-beta*rho*exp(theta*xbar))) + (6/(1-rho^2))*(rho^2/(1-beta*exp(theta*xbar))-rho^4/(1-beta*rho^2*exp(theta*xbar))) - (4/(1-rho^3))*(rho^3/(1-beta*exp(theta*xbar))-rho^6/(1-beta*rho^3*exp(theta*xbar)))+ (1/(1-rho^4))*(rho^4/(1-beta*exp(theta*xbar))-rho^8/(1-beta*rho^4*exp(theta*xbar))))*sigma4

end
