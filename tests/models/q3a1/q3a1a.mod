var c,a1,k1;
varexo e, e1;
parameters beta, gamma, phi, delta, alpha, rho, zeta, sigma, N;
alpha = 0.36;
delta = 0.1;
phi = 2;
gamma = 1;
beta = 0.99;
rho = 0.95;
zeta = 2;
sigma = 2;
N = 1;

model;
c^(-gamma)*(1+phi*zeta*(k1-k1(-1))^(zeta-1)/(2*k1(-1))) = beta*c(+1)^(-gamma)*(1+phi*zeta*(k1-k1(-1))^(zeta-1)/(2*k1(-1))+phi*(k1-k1(-1))^zeta/(2*k1(-1)^2)-delta+alpha*a1(+1)*k1^(alpha-1));
log(a1) = rho*log(a1(-1))+sigma*(e+e1);
N*c+k1+phi*(k1-k1(-1))^zeta/(2*k1)-(1-delta)*k1(-1) = a1*k1(-1)^alpha;
end;

initval;
c=1;
e=0;
a1=1;
e1=0;
k1=10;
end;

shocks;
var e; stderr 1.0;
var e1; stderr 1.0;
end;

steady;

stoch_simul(dr=cycle_reduction,order = 2, irf=0);

