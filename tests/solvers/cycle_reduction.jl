module CycleReduction

export cycle_reduction, cycle_reduction_init, cycle_reduction_core, cycle_reduction_check

function cycle_reduction_init(n)    
    A0_0 = zeros(n,n)
    Ahat1 = A0_0 
    tmp = A0_0
    id0 = 1:n
    id2 = id0+n
    return id0, id2, A0_0, Ahat1, tmp
end
    
function cycle_reduction_core(A0, A1, A2, cvg_tol, max_it, A0_0, Ahat1, tmp, id0, id2)

    it = 0
    info = [NaN, NaN]
    info[1] = 0
    crit = Inf

    A0_0 = A0
    cont = true
    while cont
        tmp = ([A0; A2]/A1)*[A0 A2]
        A1 = A1 - tmp[id0,id2] - tmp[id2,id0]
        A0 = -tmp[id0,id0]
        A2 = -tmp[id2,id2]
        Ahat1 = Ahat1 -tmp[id2,id0]
        crit = norm(A0,1)
        if crit < cvg_tol
            # keep iterating until condition on A2 is met
            if norm(A2,1) < cvg_tol
                cont = false
            end
        elseif isnan(crit) || it == max_it
            if crit < cvg_tol
                info[1] = 4
                info[2] = log(norm(A2,1))
            else
                info[1] = 3
                info[2] = log(norm(A1,1))
            end
            return info
        end        
        it = it + 1
    end

    A0 = -Ahat1\A0_0

    return info
end

function cycle_reduction_check(X,A0, A1, A2,cvg_tol)
    res = A0_0 + A1_0 * A0 + A2_0 * A0 * A0
    if (sum(sum(abs(res))) > cvg_tol)
        print("the norm of the residuals, ", res, ", compared to the tolerance criterion ",cvg_tol)
    end
    nothing
end
    
function cycle_reduction(A0, A1, A2, cvg_tol = 1e-8, maxit = 300, check = false)
    n,m = size(A0)
    if check
        A0_0 = A0
        A1_0 = A1
        A2_0 = A2
    end
    id0, id2, A0_0, Ahat1, tmp = cycle_reduction_init(n)
    info = cycle_reduction_core(A0, A1, A2, cvg_tol, maxit, A0_0, Ahat1, tmp, id0, id2)
    if check
        cycle_reduction_check(A0,A0_0,A1_0,A2_0,cvg_tol)
    end
    return A0, info
end

end
