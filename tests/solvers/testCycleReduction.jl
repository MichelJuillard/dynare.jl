using CycleReduction
n = 3
ws = CycleReductionWS(n)
a0 = [0.5 0 0; 0 0.5 0; 0 0 0];
a1 = eye(n)
a2 = [0 0 0; 0 0 0; 0 0 0.8]
x = zeros(n,n)
cycle_reduction!(x,a0,a1,a2,ws,1e-8,50)
assert(ws.info == 0)
cycle_reduction_check(x,a0,a1,a2,1e-8)
a0 = [0.5 0 0; 0 1.1 0; 0 0 0];
a1 = eye(n)
a2 = [0 0 0; 0 0 0; 0 0 0.8]
cycle_reduction!(x,a0,a1,a2,ws,1e-8,50)
assert(ws.info == 1)
a0 = [0.5 0 0; 0 0.5 0; 0 0 0];
a1 = eye(n)
a2 = [0 0 0; 0 0 0; 0 0 1.2]
cycle_reduction!(x,a0,a1,a2,ws,1e-8,50)
assert(ws.info == 2)
