
### Solution handling & optimization interface

"""Find the index value into the solution where a particular state variable is located."""
function get_solution_indexer(prob, relevant_states) 
    states = MT.get_states(prob.f.sys)
    statemap = [v => i for (i, v) in enumerate(states)]
    idxs = Int.(ModelingToolkit.varmap_to_vars(statemap, relevant_states))
    return idxs
end

"""Extract the VariableDefaultValue from a symbol or num"""
get_default(x::Num) = get_default(MT.unwrap(x))
get_default(x::Sym) = MT.getmetadata(x, MT.Symbolics.VariableDefaultValue)

"""Find the indexes into the official vector of parameters of the system behind the problem at which a subset of parameters are located."""
function get_parameter_indexer(prob, psubset)
    probpars = MT.get_ps(prob.f.sys)
    parmap = [p => i for (i, p) in enumerate(probpars)]
    idxs = Int.(ModelingToolkit.varmap_to_vars(parmap, psubset))
    return idxs
end

"""Create a function that accepts values for a subset of parameters & returns the ODE solution."""
function make_re_solver(prob, psubset, pfixed, usubset, ufixed, tbase)
    idxs_subset = get_parameter_indexer(prob, psubset)
    idxs_usubs = get_solution_indexer(prob, usubset)
    ptot_vals = map(get_default, MT.get_ps(prob.f.sys)) #HACK! Sanity check needed that psubset doesn't include pfixed variables
    utot_vals = prob.u0
    Np = length(psubset)
    if !isempty(pfixed) # Change parameter defaults
        pfixed_sub = map(x -> x[1], pfixed)
        idxs_pfixed = get_parameter_indexer(prob, pfixed_sub)
        pfixed_vals = map(x -> x[2], pfixed)
        ptot_vals[idxs_pfixed] = pfixed_vals
    end
    if !isempty(ufixed) # Change initial condition defaults
        ufixed_sub = map(x -> x[1], ufixed)
        idxs_ufixed = get_solution_indexer(prob, ufixed_sub)
        ufixed_vals = map(x -> x[2], ufixed)
        utot_vals[idxs_ufixed] = ufixed_vals
    end
    function re_solver(sub_vals::AbstractVector{T}) where T
        # Type weirdness is b/c ForwardDiff needs to stuff Duals into the vector
        # https://discourse.julialang.org/t/error-with-forwarddiff-no-method-matching-float64/41905
        ptot = Vector{T}(undef, size(ptot_vals))
        utot = Vector{T}(undef, size(utot_vals))
        ptot[1:end] .= ptot_vals
        utot[1:end] .= utot_vals
        ptot[idxs_subset] .= sub_vals[1:Np]
        utot[idxs_usubs] .= sub_vals[(Np+1):end]
        newprob = remake(prob, p = ptot, u0 = utot)
        sol = solve(newprob, Tsit5(), saveat = tbase)
        return sol
    end
    return re_solver
end

""" Solve the ODEproblem using make_re_solver"""
function re_solve(prob, psubset_map, pfixed, usubset_map, ufixed, tbase)
    psubset = map(x -> x[1], psubset_map)
    pvals = map(x -> x[2], psubset_map)
    usubset = map(x -> x[1], usubset_map)
    uvals = map(x -> x[2], usubset_map)
    vals = [pvals; uvals]
    sol = make_re_solver(prob, psubset, pfixed, usubset, ufixed, tbase)(vals)
    return sol
end

""" Solve the ODEproblem and compare the solution to the data"""
function error_plot(prob, psubset_map, pfixed, usubset_map, ufixed, dataframe)
    xdata, vdata, tdata = dataframe.x, dataframe.v, dataframe.time
    sol = re_solve(prob, psubset_map, pfixed, usubset_map, ufixed, tdata)

    plot(tdata, xdata, label="Actual x");
    pxs = plot!(sol, idxs = [x,])

    plot(tdata, vdata, label="Actual v");
    pvs = plot!(sol, idxs = [v,])

    display(plot(pxs, pvs))
    return sol
end

"""Find the values of the parameters in psubset_map (parameter/initial guess map) that minimize the loss."""
function reoptimize(prob, psubset_map, pfixed, usubset_map, ufixed, dataframe; tmax = 20, bfgs = false)
    xdata, vdata, tdata = dataframe.x, dataframe.v, dataframe.time
    init_guess = map(x -> x[2], [psubset_map; usubset_map]) 
    psubset = map(x -> x[1], psubset_map)
    usubset = map(x -> x[1], usubset_map)
    rs = make_re_solver(prob, psubset, pfixed, usubset, ufixed, tdata) 
    #Find the index map for getting at the solution variables b/c indexing by symbol is expensive
    x_indexer = only(get_solution_indexer(prob, [x])) #This is how to make it work for variables that are states of the system only
    vobsfunc = MT.build_explicit_observed_function(prob.f.sys, v) #this is more general, for observed variables or states
    function myloss(pvals)
        sol = rs(pvals)
        solreshaped = reduce(hcat, sol.u)
        xsol = solreshaped[x_indexer, :]
        vsol = map(x -> vobsfunc(x[1], sol.prob.p, x[2]), zip(sol.u, sol.t))
        xloss = sum(@. (xsol - xdata)^2)
        vloss = sum(@. (vsol - vdata)^2)
        return vloss + xloss
    end
    lower = map(x -> -1e3, init_guess)
    upper = map(x -> 1e3, init_guess)
    opt_result = Optim.optimize(myloss, lower, upper, init_guess, Fminbox(NelderMead()), Optim.Options(time_limit = tmax, iterations = 100)) #BFGS stinks, this is better
    optimized = [p => o for (p, o) in zip([psubset; usubset], opt_result.minimizer)]
    return optimized, opt_result
end