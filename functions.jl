using Distributions
using Random
using Plots
using PyPlot
using StatsBase
using StatsPlots
using Serialization
using LinearAlgebra
using LinearSolve
using IJulia
using JuMP, Gurobi
using SharedArrays

import Base.copy

ENV["GUROBI_HOME"] = "~/gurobi1001/linux64"
ENV["GRB_LICENSE_FILE"]="/home/fairleyl/gurobi1001/gurobi.lic"

const GRB_ENV = Gurobi.Env()


#set cartesian product, treating two input arrays as and bs as sets, and returning an array of tuples
function cartProd(as, bs)
    out = []
    for a in as
        for b in bs
            push!(out, [a,b])
        end
    end

    return out
end

#repeated cartesian product for sets of the form {1,2,...,M}. A is an array of arrays to be treated as sets. 
#Each A[i] has an associated cost C[i], and there can be a budget B. For example, if A=[[0,1,2],[0,1,2,3]], the tuple
# [2,3] will only be ioncluded in the final set if C[1]*2 + C[2]*3 <= B 
function repCartesianProduct(A; C = 0, B = Inf)
    N = length(A)
    if N == 1
        return [[a] for a in A[1]]
    end

    subProds = [cartProd(A[1],A[2])]
    for i in 1:(N - 2)
        newSubProds = cartProd(subProds[i], A[i+2])
        newSubProdsChecked = []
        for prod in newSubProds
            newProd = copy(prod[1])
            push!(newProd, prod[2])
            if B == Inf || sum(C[1:(i + 2)] .* newProd) <= B
                push!(newSubProdsChecked, newProd)
            end
        end

        push!(subProds, newSubProdsChecked)
    end

    return subProds[N-1]
end

#functions to construct state space
function enumerateStatesHomog(N)
    stateSpace = [[0,0]]
    for i in 1:N
        append!(stateSpace, [[j, i - j] for j in 0:i])
    end
    return stateSpace
end

function enumerateStatesESSA(A)
    B = [enumerateStatesHomog(A[i]) for i in 1:length(A)]

    return repCartesianProduct(B)
end

#functions to construct union over all action spaces to different levels of complexity
function enumerateAllActionsHomog(N)
    return [i for i in 0:N]
end

function enumerateAllActionsESSA(A)
    B = [enumerateAllActionsHomog(A[i]) for i in 1:length(A)]
    return repCartesianProduct(B)
end

function enumerateAllActionsESSA_LAS1(D)
    N = length(D)
    output = [fill(0, N)]
    for i in 1:N
        newA = fill(0, N)
        newA[i] = 1
        push!(output, newA)
    end

    return output
end

function enumerateAllActionsESSA_LAS2(D)
    N = length(D)
    output = [fill(0, N)]
    for i in 1:N
        newA = fill(0, N)
        newA[i] = 1
        for j in 1:D[i]
            push!(output, j .* newA)
        end
    end

    return output
end


#functions to construct only feasible action spaces to different levels of complexity
function enumerateFeasibleActionsHomog(s, i)
    return [i for i in 0:s[i][2]]
end

function enumerateFeasibleActionsESSA(s)
    B = [enumerateFeasibleActionsHomog(s,i) for i in 1:length(s)]
    return repCartesianProduct(B)
end

function enumerateFeasibleActionsESSA_LAS1(s)
    N = length(s)
    output = [fill(0, N)]
    for i in 1:N
        if s[i][2] > 0
            newA = fill(0, N)
            newA[i] = 1
            push!(output, newA)
        end
    end

    return output
end

function enumerateFeasibleActionsESSA_LAS2(s)
    N = length(s)
    output = [fill(0, N)]
    for i in 1:N
        if s[i][2] > 0
            newA = fill(0, N)
            newA[i] = 1
            for j in 1:s[i][2]
                push!(output, j .* newA)
            end
        end
    end

    return output
end


mutable struct problemParams
    #Misc
    N::Int64 #Number of links
    beta::Float64 #demand

    #rates
    alpha::Array{Float64} #array of deg rates
    tau::Array{Float64} #array of repair rates
    
    #Costs
    c::Array{Float64} #array of usage costs
    r::Array{Float64} #array of repair costs
    p::Float64 #penalty cost
end

function problemParams()
    return problemParams(0, 0.0, [], [], [], [], 0.0)
end

function problemParams(; N = 0, beta = 1.0, alpha = [], tau = [], c = [], r = [], p = 0.0)
    return problemParams(N, beta, alpha, tau, c, r, p)
end

function copy(probParams::problemParams)
    (; N, beta, alpha, tau, c, r, p) = probParams
    return problemParams(deepcopy(N), deepcopy(beta), deepcopy(alpha), deepcopy(tau), deepcopy(c), deepcopy(r), deepcopy(p))
end

function costRateESSA(s, probParams, D)
    (; N, alpha, beta, tau, c, p, r) = probParams

    costRate = [sum(s[i][1]*r[i] for i in 1:N), 0.0]

    healthy = false
    for i in 1:N
        if D[i] - sum(s[i]) > 0
            costRate = costRate .+ [beta*c[i], 0.0]
            healthy = true
            break
        end
    end

    if !healthy
        costRate = costRate .+ [0.0, p]
    end

    return costRate
end

function costRateESSA(s, a, probParams, D)
    (; N) = probParams
    sPrime = copy(s)
    for i in 1:N
        sPrime[i] = sPrime[i] .+ a[i]*[1,-1]
    end

    return costRateESSA(sPrime, probParams, D)
end

function expectedNextValueESSA(s,a,probParams, D, del, h)
    (; N, alpha, beta, tau, c, p, r) = probParams
    sPrime = copy(s)
    for i in 1:N
        sPrime[i] = sPrime[i] .+ (a[i] .* [1,-1])
    end

    runningTotal = [0.0,0.0]
    runningTotalProb = 0.0

    #new degradations
    for i in 1:N
        healthyI = D[i] - sum(sPrime[i])
        if healthyI > 0
            sNext = copy(sPrime)
            sNext[i] += [0,1]
            runningTotal .+= (alpha[i]*healthyI*del) .* h[sNext]
            runningTotalProb += alpha[i]*healthyI*del
        end
    end

    #repair failures
    for i in 1:N
        if sPrime[i][1] > 0
            sNext = copy(sPrime)
            sNext[i] += [-1,1]
            runningTotal .+= (sPrime[i][1]*alpha[i]*del) .* h[sNext]
            runningTotalProb += sPrime[i][1]*alpha[i]*del
        end
    end

    #repair success
    for i in 1:N
        if sPrime[i][1] > 0
            sNext = copy(sPrime)
            sNext[i] += [-1,0]
            runningTotal .+= (sPrime[i][1]*tau[i]*del) .* h[sNext]
            runningTotalProb += sPrime[i][1]*tau[i]*del
        end
    end

    return runningTotal .+ ((1 - runningTotalProb) .* h[sPrime])
end

#PI Action Construction
function piActionESSA(s, actionSpace, h, probParams, del, D)
    (; N, alpha, beta, tau, c, p, r) = probParams
    if sum(s[i][2] for i in 1:N) == 0
        optA = fill(0, N)
        optH = (costRateESSA(s, probParams, D) .* del) .+ expectedNextValueESSA(s, optA, probParams, D, del, h)
        return optA, optH
    end

    optA = fill(0, N)
    optH = (costRateESSA(s, probParams, D) .* del) .+ expectedNextValueESSA(s, optA, probParams, D, del, h)

    for testA in actionSpace
        testH = (costRateESSA(s, testA, probParams, D) .* del) .+ expectedNextValueESSA(s, testA, probParams, D, del, h)
        if sum(testH) < sum(optH)
            optH = testH
            optA = testA
        end
    end

    return optA, optH
end

function piPolicyESSA(h, actionSpaces, probParams, del, D)
    policy = Dict()
    stateSpace = enumerateStatesESSA(D)
    for s in stateSpace
        policy[s] = piActionESSA(s, actionSpaces[s], h, probParams, del, D)[1]
    end
    policy = policySequencer(policy)
    return policy
end
    
function rpeESSA(probParams, D, policy, epsilon; nMax = 0, delScale = 1.0, printProgress = false, modCounter = 1000)
    (; N, alpha, beta, tau, c, p, r) = probParams
    del = 1/(delScale*(sum(D[i]*(alpha[i] + tau[i]) for i in 1:N)))
    h = Dict()
    w = Dict()

    stateSpace = enumerateStatesESSA(D)
    for s in stateSpace
        h[s] = [0.0, 0.0]
        w[s] = [0.0, 0.0]
    end
    s0  = fill([0,0], N)
    n = 0
    deltas = Dict()
    delta = 0.0

    while true
        n += 1
        for s in stateSpace
            a = policy[s]
            w[s] = (costRateESSA(s, a, probParams, D) .* del) .+ expectedNextValueESSA(s, a, probParams, D, del, h)
        end

        #calculate relative values and delta
        for s in stateSpace
            update = w[s] .- w[s0]
            deltas[s] = update .- h[s]
            h[s] = update
        end

        deltas1 = [v[1] for v in values(deltas)]
        deltas2 = [v[2] for v in values(deltas)]
        delta1 = maximum(deltas1) - minimum(deltas1)
        delta2 = maximum(deltas2) - minimum(deltas2)
        #stopping condition
        if max(delta1,delta2) < epsilon || n == nMax
            break
        end

        if printProgress && n%modCounter == 0
            println(n)
        end
    end
    a0 = fill(0, N)
    g = (costRateESSA(s0, a0, probParams, D) .* del) .+ expectedNextValueESSA(s0, a0, probParams, D, del, h) - h[s0]

    return g/del, h, n, delta
end

function rpiESSA(probParams, D, hIn, epsilon; nMax = 0, delScale = 1, printProgress = false, modCounter = 1000, actionType = "las1")
    (; N, alpha, beta, tau, c, p, r) = probParams
    del = 1/(delScale*(sum(D[i]*(alpha[i] + tau[i]) for i in 1:N)))
    stateSpace = enumerateStatesESSA(D)
    actionSpaces = Dict()
    if actionType == "las1"
        for s in stateSpace
            actionSpaces[s] = enumerateFeasibleActionsESSA_LAS1(s)
        end
    elseif actionType == "las2"
        for s in stateSpace
            actionSpaces[s] = enumerateFeasibleActionsESSA_LAS2(s)
        end
    elseif actionType == "full"
        for s in stateSpace
            actionSpaces[s] = enumerateFeasibleActionsESSA(s)
        end
    else
        throw(DomainError(actionType, "Invalid actionType"))
    end

    policy = piPolicyESSA(hIn, actionSpaces, probParams, del, D)
    output = rpeESSA(probParams, D, policy, epsilon; nMax = nMax, delScale = delScale, printProgress = printProgress, modCounter = modCounter)
    return output[1], output[2], output[3], policy 
end

function rviESSA(probParams, D, epsilon; nMax = 0, delScale = 1, printProgress = true, modCounter = 1000, actionType = "las1")
    (; N, alpha, beta, tau, c, p, r) = probParams
    del = 1/(delScale*(sum(D[i]*(alpha[i] + tau[i]) for i in 1:N)))
    stateSpace = enumerateStatesESSA(D)
    actionSpaces = Dict()
    if actionType == "las1"
        for s in stateSpace
            actionSpaces[s] = enumerateFeasibleActionsESSA_LAS1(s)
        end
    elseif actionType == "las2"
        for s in stateSpace
            actionSpaces[s] = enumerateFeasibleActionsESSA_LAS2(s)
        end
    elseif actionType == "full"
        for s in stateSpace
            actionSpaces[s] = enumerateFeasibleActionsESSA(s)
        end
    else
        throw(DomainError(actionType, "Invalid actionType"))
    end
    #println(actionSpaces)
    h = Dict()
    w = Dict()
    policy = Dict()
    
    for s in stateSpace
        h[s] = [0.0, 0.0]
        w[s] = [0.0, 0.0]
        policy[s] = fill(0, N)
    end
    s0  = fill([0,0], N)
    n = 0
    
    deltas = Dict()
    delta1 = 0.0
    delta2 = 0.0
    delta = 0.0
    #do until max iterations met or epsilon convergence
    while true
        n = n + 1
        #find updates for every state
        for s in stateSpace
            as = actionSpaces[s]
            policy[s],w[s] = piActionESSA(s, as, h, probParams, del, D)
        end
        
        #calculate relative values and delta
        for s in stateSpace
            update = w[s] .- w[s0]
            deltas[s] = update .- h[s]
            h[s] = update
        end
        
        #println(h[[[1,0],[2,0],[1,0]]])
        deltas1 = [v[1] for v in values(deltas)]
        deltas2 = [v[2] for v in values(deltas)]
        delta1 = maximum(deltas1) - minimum(deltas1)
        delta2 = maximum(deltas2) - minimum(deltas2)
        delta = max(delta1,delta2)
        #stopping condition
        if delta < epsilon || n == nMax
            break
        end

        if printProgress && n%modCounter == 0
            println(n)
        end
    end
    a0 = fill(0, N)
    g = (costRateESSA(s0, a0, probParams, D) .* del) .+ expectedNextValueESSA(s0, a0, probParams, D, del, h) .- h[s0]

    return g ./ del, h, n, delta
end

function fullyActiveESSA(D)
    N = length(D)
    stateSpace = enumerateStatesESSA(D)
    policy = Dict()
    for s in stateSpace
        a = fill(0, N)
        for i in 1:N
            a[i] = s[i][2]
        end
        policy[s] = a
    end
    return policy
end 

function postActionState(s, a)
    N = length(a)
    sPrime = copy(s)
    for i in 1:N
        sPrime[i] = sPrime[i] .+ (a[i] .* [1, -1])
    end
    return sPrime
end

function actionSequencer(s, policy)
    N = length(s)
    aOut = fill(0, N)
    sPrime = copy(s)
    a = policy[sPrime]
    while a != fill(0, N)
        aOut = aOut .+ a
        sPrime = postActionState(sPrime, a)
        a = policy[sPrime]
    end

    return aOut
end

function policySequencer(policy)
    newPolicy = Dict()
    stateSpace = keys(policy)
    for s in stateSpace
        newPolicy[s] = actionSequencer(s, policy)
    end

    return newPolicy
end

    


################################################
#STATIC STUFF###################################
################################################
function dcpIntLinearisationUnconstrained(probParams::problemParams; numLinks = Inf, M = 1.0, timeLimit = 600.0)
    (;N, alpha, tau, c, p, r) = probParams
    ps = tau ./ (tau + alpha)
    qs = 1 .- ps

    upper = upperBoundOnLinkCopies(probParams)
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "IntegralityFocus", 1)
    set_optimizer_attribute(model, "NumericFocus", 3)
    set_optimizer_attribute(model, "Quad", 1)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    set_optimizer_attribute(model, "OptimalityTol", 1e-9)
    set_optimizer_attribute(model, "MarkowitzTol", 0.999)
    
    
    indices = [(i,j) for i in 1:N for j in 1:upper[i]]
    indices2 = [(i,j) for i in 1:N for j in 1:upper[i] if j >= 2]
    indices3 = [(i,j) for i in 1:N for j in 1:upper[i] if j < upper[i]]
    @variable(model, x[indices], Bin)
    @variables(model, begin
        y[indices] >=0
        z[indices] >=0
        end
    )

    @constraint(model, eq, y[(1,1)] + z[(1,1)]  == 1)
    @constraint(model, eq2[i in indices2], M*y[i] + M*z[i] - M*z[i .- (0,1)] == 0.0)
    @constraint(model, eq3[i in 2:N], M*y[(i,1)] + M*z[(i,1)] == M*z[(i-1, upper[i - 1])])

    @constraint(model, ineq1[i in indices], M*y[i] <= M*ps[i[1]]*x[i])
    @constraint(model, ineq2[i in indices2], M*y[i] <= M*ps[i[1]]*z[(i .- (0,1))])
    @constraint(model, ineq3[i in 2:N], M*y[(i,1)] <= M*ps[i]*z[(i-1, upper[i-1])])

    @constraint(model, [i in indices3], x[i] >= x[i .+ (0,1)])
    @constraint(model, [i in indices2], M*z[i] <= M*z[i .- (0,1)])
    @constraint(model, [i in 2:N], M*z[(i,1)] <= M*z[(i-1,upper[i-1])])
    
    if numLinks < Inf
        @constraint(model, sum(x[(i,j)] for (i,j) in indices) == numLinks)
    end

    #create objective
    @objective(model,
    Min,
    sum(M*r[i]*qs[i]*x[(i,j)] + M*c[i]*y[(i,j)] for (i,j) in indices) + M*p*z[(N, upper[N])])

    optimize!(model)

    try
        return objective_value(model), model, x, y, z
    catch
        return "No Solution", model, x, y, z
    end
end

#ε-δ-DOP-PC 
#Input: Problem parameters, probLim is ε, M just scales variables (not needed), C and w are costs and weights, B and W are respective budgets, epsilon is actually δ
#Output: the model, operational cost (corrected to ignore 1+δ term), LFR, and optimal x y z values  
function dcpIntConstrainedFailure(probParams::problemParams; probLim = 0.0, M = 1.0, C = 0.0, B = Inf, w = 0.0, W = Inf, epsilon = 1.0e-1, timeLimit = 600.0)
    (;N, alpha, tau, c, p, r) = probParams
    ps = tau ./ (tau + alpha)
    qs = 1 .- ps

    upper = max.(ceil.(probLim ./ log.(qs)), 1) #upper = upperBoundOnLinkCopies(probParams)
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "IntegralityFocus", 1)
    set_optimizer_attribute(model, "NumericFocus", 3)
    set_optimizer_attribute(model, "Quad", 1)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    set_optimizer_attribute(model, "OptimalityTol", 1e-9)
    set_optimizer_attribute(model, "MarkowitzTol", 0.999)
    set_optimizer_attribute(model, "TimeLimit", timeLimit)

    indices = [(i,j) for i in 1:N for j in 1:upper[i]]
    indices2 = [(i,j) for i in 1:N for j in 1:upper[i] if j >= 2]
    indices3 = [(i,j) for i in 1:N for j in 1:upper[i] if j < upper[i]]

    @variable(model, x[indices], Bin)
    @variables(model, begin
        y[indices] >=0
        z[indices] >=0
        end
    )
    
    @constraint(model, eq, y[(1,1)] + z[(1,1)]  == 1)
    @constraint(model, eq2[i in indices2], M*y[i] + M*z[i] - M*z[i .- (0,1)] == 0.0)
    @constraint(model, eq3[i in 2:N], M*y[(i,1)] + M*z[(i,1)] == M*z[(i-1, upper[i - 1])])

    @constraint(model, ineq1[i in indices], M*y[i] <= M*ps[i[1]]*x[i])
    @constraint(model, ineq2[i in indices2], M*y[i] <= M*ps[i[1]]*z[(i .- (0,1))])
    @constraint(model, ineq3[i in 2:N], M*y[(i,1)] <= M*ps[i]*z[(i-1, upper[i-1])])

    @constraint(model, [i in indices3], x[i] >= x[i .+ (0,1)])

    @constraint(model, [i in indices2], z[i] <= z[i .- (0,1)])
    @constraint(model, [i in 2:N], z[(i,1)] <= z[(i-1,upper[i-1])])

    if B < Inf
        @constraint(model, sum(C[i]*x[(i,j)] for (i,j) in indices) <= B)
    end 

    if W < Inf
        @constraint(model, sum(w[i]*x[(i,j)] for (i,j) in indices) <= W)
    end

    #create objective
    @objective(model,
    Min,
    sum(r[i]*qs[i]*x[(i,j)] + c[i]*y[(i,j)] for (i,j) in indices) + (1 + epsilon)*maximum(c)*z[(N, upper[N])])
    @constraint(model, sum(log(qs[i])*x[(i,j)] for (i,j) in indices) <= probLim)
    optimize!(model)

    try
        objVal = objective_value(model) - (1 + epsilon)*maximum(c)*value(z[(N, upper[N])])
        logFailProb = sum(log(qs[i])*value(x[(i,j)]) for (i,j) in indices)
        return model, objVal, logFailProb, x, y, z 
    catch
        return model, "No Solution", "No Solution", x, y, z  
    end
end



function dcpIntMinCostConstrainedFailure(probParams::problemParams, C, probLim)
    (;N, alpha, tau, c, p, r) = probParams
    ps = tau ./ (tau + alpha)
    qs = 1 .- ps

    upper = max.(ceil.(probLim ./ log.(qs)), 1) #upper = upperBoundOnLinkCopies(probParams)
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "IntegralityFocus", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    #set_optimizer_attribute(model, "Quad", 1)
    #set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    #set_optimizer_attribute(model, "OptimalityTol", 1e-9)
    #set_optimizer_attribute(model, "MarkowitzTol", 0.999)

    @variable(model, x[1:N] >= 0, Int)

    @constraint(model, sum(log(qs[i])*x[i] for i in 1:N) <= probLim)

    @objective(model,
    Min,
    sum(C[i]*x[i] for i in 1:N))

    optimize!(model)
    return model, objective_value(model), x 
end
    
function dcpIntMinMixCostWeightConstrainedFailure(probParams::problemParams, C, w, gamma, probLim)
    (;N, alpha, tau, c, p, r) = probParams
    ps = tau ./ (tau + alpha)
    qs = 1 .- ps

    upper = max.(ceil.(probLim ./ log.(qs)), 1) #upper = upperBoundOnLinkCopies(probParams)
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "IntegralityFocus", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    #set_optimizer_attribute(model, "Quad", 1)
    #set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    #set_optimizer_attribute(model, "OptimalityTol", 1e-9)
    #set_optimizer_attribute(model, "MarkowitzTol", 0.999)

    @variable(model, x[1:N] >= 0, Int)

    @constraint(model, sum(log(qs[i])*x[i] for i in 1:N) <= probLim)

    @objective(model,
    Min,
    sum((gamma*C[i] + (1 - gamma)w[i])*x[i] for i in 1:N))

    optimize!(model)
    return model, sum(C[i]*value(x[i]) for i in 1:N), sum(w[i]*value(x[i]) for i in 1:N), x 
end

function dcpIntBinToVar(x)
    indices = keys(x)
    N = maximum([i[1][1] for i in indices])
    #print(N)
    count = fill(0, N)
    for i in indices
        count[i[1][1]] = count[i[1][1]] + value.(x[i])
    end
    return count
end

function dcpIntMinFailureConstrainedCost(probParams::problemParams, C, B; w = 0.0, W = Inf)
    (;N, alpha, tau, c, p, r) = probParams
    ps = tau ./ (tau + alpha)
    qs = 1 .- ps
    upper = max.(floor.(B ./ C), 1)
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "IntegralityFocus", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    #set_optimizer_attribute(model, "Quad", 1)
    #set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    #set_optimizer_attribute(model, "OptimalityTol", 1e-9)
    #set_optimizer_attribute(model, "MarkowitzTol", 0.999)

    @variable(model, x[1:N] >= 0, Int)

    @constraint(model, sum(C[i]*x[i] for i in 1:N) <= B)

    if W < Inf
        @constraint(model, sum(w[i]*x[i] for i in 1:N) <= W)
    end 

    @objective(model,
    Min,
    sum(log(qs[i])*x[i] for i in 1:N))

    optimize!(model)

    return model, objective_value(model), x
end

function neighbourhood(s)
    neighbourhood = []
    N = length(s)
    for i in 1:N
        sRepSucc = copy(s)
        sRepFail = copy(s)
        sDeg = copy(s)
        sRepSucc[i] = sRepSucc[i] .+ [0,-1]
        sRepFail[i] = sRepFail[i] .+ [1,-1]
        sDeg[i] = sDeg[i] .+ [1,0]
        append!(neighbourhood, [sRepSucc, sRepFail, sDeg])
    end

    return neighbourhood
end 

function q(probParams, D)
    #STUFF
    (;N, alpha, tau, c, p, r) = probParams
    q = Dict()
    stateSpace = enumerateStatesESSA(D)
    for s in stateSpace
        actionSpace = enumerateFeasibleActionsESSA(s)
        for a in actionSpace
            sPost = postActionState(s,a)
            #nHood = neighbourhood(sPost)
            for sPrime in stateSpace
                q[s,a,sPrime] = 0.0
            end

            total = 0.0
            for i in 1:N
                sRepSucc = deepcopy(sPost)
                sRepFail = deepcopy(sPost)
                sDeg = deepcopy(sPost)
                sRepSucc[i] = sRepSucc[i] .+ [-1,0]
                sRepFail[i] = sRepFail[i] .+ [-1,1]
                sDeg[i] = sDeg[i] .+ [0,1]

                sDiff = (s != sRepFail)
                q[s,a,sRepSucc] = tau[i]*sPost[i][1]
                q[s,a,sRepFail] = sDiff*alpha[i]*sPost[i][1]
                q[s,a,sDeg] = (D[i] - sum(sPost[i]))*alpha[i]

                total += (tau[i] + sDiff*alpha[i])*sPost[i][1] + (D[i] - sum(sPost[i]))*alpha[i]
            end

            q[s,a,s] = -total
        end
    end 
    return q
end

function qWithPolicy(probParams, stateSpace, D, policy)
    #STUFF
    (;N, alpha, tau, c, p, r) = probParams
    q = Dict()
    for s in stateSpace
        a = policy[s]
        
        sPost = postActionState(s,a)
        for sPrime in stateSpace
            q[s,sPrime] = 0.0
        end

        total = 0.0
        for i in 1:N
            sRepSucc = deepcopy(sPost)
            sRepFail = deepcopy(sPost)
            sDeg = deepcopy(sPost)
            sRepSucc[i] = sRepSucc[i] .+ [-1,0]
            sRepFail[i] = sRepFail[i] .+ [-1,1]
            sDeg[i] = sDeg[i] .+ [0,1]

            sDiff = (s != sRepFail)
            q[s,sRepSucc] = tau[i]*sPost[i][1]
            q[s,sRepFail] = sDiff*alpha[i]*sPost[i][1]
            q[s,sDeg] = (D[i] - sum(sPost[i]))*alpha[i]

            total += (tau[i] + sDiff*alpha[i])*sPost[i][1] + (D[i] - sum(sPost[i]))*alpha[i]
        end

        q[s,s] = -total
        
    end 
    return q
end

function policyFromFreqs(f, stateSpace; actionType = "full")
    actionSpaceFunc = print #placeholder function
    if actionType == "full"
        actionSpaceFunc = enumerateFeasibleActionsESSA
    elseif actionType == "las1"
        actionSpaceFunc = enumerateFeasibleActionsESSA_LAS1
    elseif actionType == "las2"
        actionSpaceFunc = enumerateFeasibleActionsESSA_LAS2
    else
        throw(DomainError(actionType, "Invalid actionType"))
    end

    total = 0.0
    s0 = fill([0,1], 4)
    policy = Dict()
    for s in stateSpace 
        actionSpace = actionSpaceFunc(s)
        fs = [f[[s,a]] for a in actionSpace]
        if s == s0
            #println(fs)
        end
        
        index = argmax(fs)
        a = actionSpace[index]
        if maximum(fs) > 0.0
            #println(maximum(fs))
            #println(s)
            #println(a)
            total += maximum(fs)
        end
        policy[s] = a
    end
    #println(total)
    return policy
       
end 


function q2(probParams, D; goalTime = Inf, actionType = "full")
    #STUFF
    if time() > goalTime 
        return Dict()
    end

    (;N, alpha, tau, c, p, r) = probParams
    q = Dict()
    stateSpace = enumerateStatesESSA(D)
    for s in stateSpace
        actionSpace = []
        if actionType == "full"
            actionSpace = enumerateFeasibleActionsESSA(s)
        elseif actionType == "las1"
            actionSpace = enumerateFeasibleActionsESSA_LAS1(s)
        elseif actionType == "las2"
            actionSpace = enumerateFeasibleActionsESSA_LAS2(s)
        else
            throw(DomainError(actionType, "Invalid actionType"))
        end

        for a in actionSpace
            sPost = postActionState(s,a)
            nHood = neighbourhood(sPost)
            for sPrime in nHood
                q[s,a,sPrime] = 0.0
            end

            total = 0.0
            for i in 1:N
                sRepSucc = deepcopy(sPost)
                sRepFail = deepcopy(sPost)
                sDeg = deepcopy(sPost)
                sRepSucc[i] = sRepSucc[i] .+ [-1,0]
                sRepFail[i] = sRepFail[i] .+ [-1,1]
                sDeg[i] = sDeg[i] .+ [0,1]

                sDiff = (s != sRepFail)
                q[s,a,sRepSucc] = tau[i]*sPost[i][1]
                q[s,a,sRepFail] = sDiff*alpha[i]*sPost[i][1]
                q[s,a,sDeg] = (D[i] - sum(sPost[i]))*alpha[i]

                total += (tau[i] + sDiff*alpha[i])*sPost[i][1] + (D[i] - sum(sPost[i]))*alpha[i]
            end

            q[s,a,s] = -total
            if time() >= goalTime
                return q
            end
        end    
    end 
    return q
end


function reverseNeighbourhood(s, D; actionType = "full")
    N = length(s)
    postStateHood = []
    saHood = []
    for i in 1:N
        #repair success
        if D[i] - s[i][1] - s[i][2] > 0
            sPost = deepcopy(s)
            sPost[i][1] += 1
            push!(postStateHood, sPost)
        end
        #repair fail or new degradation
        if s[i][2] > 0
            sPost1 = deepcopy(s)
            sPost2 = deepcopy(s)
            sPost1[i] .+= [1,-1]
            sPost2[i] .+= [0,-1]
            append!(postStateHood, [sPost1, sPost2])
        end 
    end
    for sPost in postStateHood
        auxState = [[0,sPost[i][1]] for i in 1:N]
        actionSpace = []
        if actionType == "full"
            actionSpace = enumerateFeasibleActionsESSA(auxState)
        elseif actionType == "las1"
            actionSpace = enumerateFeasibleActionsESSA_LAS1(auxState)
        elseif actionType == "las2"
            actionSpace = enumerateFeasibleActionsESSA_LAS2(auxState)
        else
            throw(DomainError(actionType, "Invalid actionType"))
        end

        for a in actionSpace
            sPre = deepcopy(sPost)
            for i in 1:N
                sPre[i] += a[i].*[-1,1]
            end
            if sPre != s
                push!(saHood, [sPre, a])
            end
        end
    end

    return saHood
end


function mdpDesignLP(probParams::problemParams, C, B, w, W; minProb = 0.0, speak = false, careful = true)
    #Get component parameters
    (;N, alpha, tau, c, p, r) = probParams

    #Find maximum mumber of each component according to constraints
    upper = max.(min.(floor.(B ./ C), floor.(W ./ w)), 1)

    #start LP model and set attributes
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    if careful
        set_optimizer_attribute(model, "IntegralityFocus", 1)
        set_optimizer_attribute(model, "NumericFocus", 3)
        set_optimizer_attribute(model, "Quad", 1)
        set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
        set_optimizer_attribute(model, "OptimalityTol", 1e-9)
        set_optimizer_attribute(model, "MarkowitzTol", 0.999)
    end

    #find maximum state space and state-action space, ignoring null action in worst state
    stateSpace = enumerateStatesESSA(upper)
    stateActionSpace = []
    for s in stateSpace
        actionSpace = enumerateFeasibleActionsESSA(s)
        for a in actionSpace
            #if sum(s[i][2] for i in 1:N) < sum(upper) || a != fill(0, N)
            push!(stateActionSpace, [s, a])
            #end
        end
    end

    if speak
        println("State-Action Space Constructed")
        print(length(stateActionSpace))
        println(" variables")
    end

    # sasHealth = []
    # for sa in stateActionSpace
    #     s = sa[1]
    #     for i in 1:N
    #         if upper[i] - s[i][2] > 0
    #             push!(sasHealth, sa)
    #             break
    #         end
    #     end
    # end


    #define state-action frequency and binary design variables 
    indices = [(i,j) for i in 1:N for j in 1:upper[i]]
    @variable(model, f[stateActionSpace] >= 0, start = 0.0)
    @variable(model, x[indices], Bin, start = 0)
    #@variable(model, x[indices] >= 0, start = 0.0) #linear relaxation

    #starting solution (install and repair one copy of component 1)
    # sFail = [[0, upper[i]] for i in 1:N]
    # aFail = fill(0, N)
    # aFail[1] = 1
    # set_start_value(x[(1,1)], 1.0)
    # set_start_value(f[[sFail,aFail]], alpha[1]/(tau[1] + alpha[1]))
    # set_start_value(f[[postActionState(sFail,aFail),fill(0, N)]], tau[1]/(tau[1] + alpha[1]))

    #construct matrix of infinitesimal generator and instant costs
    qMatrix = q2(probParams, upper)
    cMatrix = Dict()
    for sa in stateActionSpace
        cMatrix[sa] = costRateESSA(sa[1],sa[2],probParams, upper)
    end

    if speak
        println("q and c matricies loaded")
    end

    #for each state, add equilibrium constraint (ignoring null action for worst state)
    # for s in stateSpace
    #     actionSpace = enumerateFeasibleActionsESSA(s)
    #     if sum(s[i][2] for i in 1:N) == sum(upper)
    #         big = length(actionSpace)
    #         actionSpace = actionSpace[2:big]
    #     end
        
    #     @constraint(model, sum(qMatrix[s,a,s]*f[[s,a]] for a in actionSpace) + sum(qMatrix[s,a,sPrime]*f[[s,a]] for a in actionSpace for sPrime in neighbourhood(postActionState(s,a))) == 0.0)
    # end

    for sPrime in stateSpace
        feasStateActionSpace = reverseNeighbourhood(sPrime, upper)
        feasActionSpace = enumerateFeasibleActionsESSA(sPrime)
        #@constraint(model, sum(qMatrix[sa[1],sa[2],sPrime]*f[sa] for sa in stateActionSpace) == 0)
        @constraint(model, sum(qMatrix[sPrime,a,sPrime]*f[[sPrime,a]] for a in feasActionSpace) + sum(qMatrix[sa[1],sa[2],sPrime]*f[sa] for sa in feasStateActionSpace) == 0)
    end
    

    #probabilities sum to one
    @constraint(model, sum(f[sa] for sa in stateActionSpace) == 1.0)

    #@constraint(model, sum(f[sa] for sa in sasHealth) >= minProb)

    #ensure freqs are 0, ie states are inaccessible, if a corresponding component is not installed   
    for i in 1:N
        for j in 1:upper[i]
            @constraint(model, sum(f[sa] for sa in stateActionSpace if upper[i] - sa[1][i][2] >= j) <= x[(i,j)])
        end
    end

    #cost and weight constraints
    @constraint(model, sum(C[i]*x[(i,j)] for i in 1:N for j in 1:upper[i]) <= B)
    @constraint(model, sum(w[i]*x[(i,j)] for i in 1:N for j in 1:upper[i]) <= W)

    #symmetry breaking constraints
    for i in 1:N
        for j in 1:(upper[i] - 1)
            @constraint(model, x[(i,j)] >= x[(i,j + 1)])
        end
    end

    if speak
        println("Constraints Loaded")
    end

    @objective(model,
    Min,
    sum(sum(cMatrix[sa])*f[sa] for sa in stateActionSpace))

    optimize!(model)

    opCost = sum(cMatrix[sa][1]*value(f[sa]) for sa in stateActionSpace)
    reliability = sum(cMatrix[sa][2]*value(f[sa]) for sa in stateActionSpace)/p
    return model, opCost, reliability , f, x
end

function mdpDesignLP_MultiP(probParams::problemParams, C, B, w, W, ps; speak = false, careful = true, timeLimit = Inf, memLim = 12, actionType = "full")
    goalTime = Inf
    if timeLimit < Inf
        goalTime = time() + timeLimit
    end

    #Get component parameters
    (;N, alpha, tau, c, p, r) = probParams
    thisProbParams = copy(probParams)
    thisProbParams.p = 1.0

    #Find maximum mumber of each component according to constraints
    upper = max.(min.(floor.(B ./ C), floor.(W ./ w)), 1)

    #start LP model and set attributes
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    if careful
        set_optimizer_attribute(model, "IntegralityFocus", 1)
        set_optimizer_attribute(model, "NumericFocus", 3)
        set_optimizer_attribute(model, "Quad", 1)
        set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
        set_optimizer_attribute(model, "OptimalityTol", 1e-9)
        set_optimizer_attribute(model, "MarkowitzTol", 0.999)
    end

    set_optimizer_attribute(model, "SoftMemLimit", memLim)
    #find maximum state space and state-action space, ignoring null action in worst state
    stateSpace = enumerateStatesESSA(upper)
    stateActionSpace = []
    for s in stateSpace
        actionSpace = []
        if actionType == "full"
            actionSpace = enumerateFeasibleActionsESSA(s)
        elseif actionType == "las1"
            actionSpace = enumerateFeasibleActionsESSA_LAS1(s)
        elseif actionType == "las2"
            actionSpace = enumerateFeasibleActionsESSA_LAS2(s)
        else
            throw(DomainError(actionType, "Invalid actionType"))
        end
        for a in actionSpace
            #if sum(s[i][2] for i in 1:N) < sum(upper) || a != fill(0, N)
            push!(stateActionSpace, [s, a])
            #end
        end
    end

    if goalTime < time()
        return "TimeOut", [], [],[]
    end

    if speak
        println("State-Action Space Constructed")
        print(length(stateActionSpace))
        println(" variables")
    end


    #define state-action frequency and binary design variables 
    indices = [(i,j) for i in 1:N for j in 1:upper[i]]
    @variable(model, f[stateActionSpace] >= 0, start = 0.0)
    @variable(model, x[indices], Bin, start = 0)
    #@variable(model, x[indices] >= 0, start = 0.0) #linear relaxation

    if goalTime < time()
        return "TimeOut", [], [],[]
    end

    #construct matrix of infinitesimal generator and instant costs
    qMatrix = q2(probParams, upper; goalTime = goalTime, actionType = actionType)

    if goalTime < time()
        return "TimeOut", [], [],[]
    end

    cMatrix = Dict()
    for sa in stateActionSpace
        cMatrix[sa] = costRateESSA(sa[1],sa[2],thisProbParams, upper)
    end

    if goalTime < time()
        return "TimeOut", [], [],[]
    end

    if speak
        println("q and c matricies loaded")
    end

    for sPrime in stateSpace
        feasStateActionSpace = reverseNeighbourhood(sPrime, upper; actionType = actionType)
        feasActionSpace = []
        if actionType == "full"
            feasActionSpace = enumerateFeasibleActionsESSA(sPrime)
        elseif actionType == "las1"
            feasActionSpace = enumerateFeasibleActionsESSA_LAS1(sPrime)
        elseif actionType == "las2"
            feasActionSpace = enumerateFeasibleActionsESSA_LAS2(sPrime)
        else
            throw(DomainError(actionType, "Invalid actionType"))
        end
        #@constraint(model, sum(qMatrix[sa[1],sa[2],sPrime]*f[sa] for sa in stateActionSpace) == 0)
        @constraint(model, sum(qMatrix[sPrime,a,sPrime]*f[[sPrime,a]] for a in feasActionSpace) + sum(qMatrix[sa[1],sa[2],sPrime]*f[sa] for sa in feasStateActionSpace) == 0)
        if goalTime < time()
            return "TimeOut", [], [],[]
        end
    end
    

    #probabilities sum to one
    @constraint(model, sum(f[sa] for sa in stateActionSpace) == 1.0)

    #ensure freqs are 0, ie states are inaccessible, if a corresponding component is not installed   
    for i in 1:N
        for j in 1:upper[i]
            @constraint(model, sum(f[sa] for sa in stateActionSpace if upper[i] - sa[1][i][2] >= j) <= x[(i,j)])
        end
    end

    #cost and weight constraints
    @constraint(model, sum(C[i]*x[(i,j)] for i in 1:N for j in 1:upper[i]) <= B)
    @constraint(model, sum(w[i]*x[(i,j)] for i in 1:N for j in 1:upper[i]) <= W)

    #symmetry breaking constraints
    for i in 1:N
        for j in 1:(upper[i] - 1)
            @constraint(model, x[(i,j)] >= x[(i,j + 1)])
        end
    end

    if speak
        println("Constraints Loaded")
    end

    opCosts = []
    reliabilities = []
    xs = []

    for p in ps
        t = goalTime - time()
        if  t < 0
            return "TimeOut", opCosts, reliabilities, xs
        else
            set_time_limit_sec(model, t)
        end

        @objective(model,
        Min,
        sum((cMatrix[sa][1] + p*cMatrix[sa][2])*f[sa] for sa in stateActionSpace))


        optimize!(model)

        y = all_variables(model)
        y_solution = value.(y)
        set_start_value.(y, y_solution)

        opCost = sum(cMatrix[sa][1]*value(f[sa]) for sa in stateActionSpace)
        reliability = log(sum((cMatrix[sa][2])*value(f[sa]) for sa in stateActionSpace)) 

        push!(opCosts, opCost)
        push!(reliabilities, reliability)
        try
            push!(xs, value.(x))
        catch err
            break
        end
    end

    return "Success", opCosts, reliabilities, xs
end

function mdpNonDesignLP_MultiP(probParams::problemParams, D, ps; speak = false, careful = true, timeLimit = Inf, memLim = 12, actionType = "full", M = 1.0)
    goalTime = Inf
    if timeLimit < Inf
        goalTime = time() + timeLimit
    end

    #Get component parameters
    (;N, alpha, tau, c, p, r) = probParams
    thisProbParams = copy(probParams)
    thisProbParams.p = 1.0

    #start LP model and set attributes
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    if careful
        set_optimizer_attribute(model, "IntegralityFocus", 1)
        set_optimizer_attribute(model, "NumericFocus", 3)
        set_optimizer_attribute(model, "Quad", 1)
        set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
        set_optimizer_attribute(model, "OptimalityTol", 1e-9)
        set_optimizer_attribute(model, "MarkowitzTol", 0.999)
    end

    set_optimizer_attribute(model, "SoftMemLimit", memLim)
    #find maximum state space and state-action space, ignoring null action in worst state
    stateSpace = enumerateStatesESSA(D)
    stateActionSpace = []
    for s in stateSpace
        actionSpace = []
        if actionType == "full"
            actionSpace = enumerateFeasibleActionsESSA(s)
        elseif actionType == "las1"
            actionSpace = enumerateFeasibleActionsESSA_LAS1(s)
        elseif actionType == "las2"
            actionSpace = enumerateFeasibleActionsESSA_LAS2(s)
        else
            throw(DomainError(actionType, "Invalid actionType"))
        end
        for a in actionSpace
            #if sum(s[i][2] for i in 1:N) < sum(upper) || a != fill(0, N)
            push!(stateActionSpace, [s, a])
            #end
        end
    end

    if goalTime < time()
        return "TimeOut", [], []
    end

    if speak
        println("State-Action Space Constructed")
        print(length(stateActionSpace))
        println(" variables")
    end


    #define state-action frequency and binary design variables 
    @variable(model, f[stateActionSpace] >= 0, start = 0.0)
    #@variable(model, x[indices] >= 0, start = 0.0) #linear relaxation

    if goalTime < time()
        return "TimeOut", [], []
    end

    #construct matrix of infinitesimal generator and instant costs
    qMatrix = q2(probParams, D; goalTime = goalTime, actionType = actionType)

    if speak
        println("q loaded")
    end 

    if goalTime < time()
        return "TimeOut", [], []
    end

    cMatrix = Dict()
    for sa in stateActionSpace
        cMatrix[sa] = costRateESSA(sa[1],sa[2],thisProbParams, D)
    end

    if goalTime < time()
        return "TimeOut", [], []
    end

    if speak
        println("c loaded")
    end

    for sPrime in stateSpace
        feasStateActionSpace = reverseNeighbourhood(sPrime, D; actionType = actionType)
        feasActionSpace = []
        if actionType == "full"
            feasActionSpace = enumerateFeasibleActionsESSA(sPrime)
        elseif actionType == "las1"
            feasActionSpace = enumerateFeasibleActionsESSA_LAS1(sPrime)
        elseif actionType == "las2"
            feasActionSpace = enumerateFeasibleActionsESSA_LAS2(sPrime)
        else
            throw(DomainError(actionType, "Invalid actionType"))
        end
        #@constraint(model, sum(qMatrix[sa[1],sa[2],sPrime]*f[sa] for sa in stateActionSpace) == 0)
        @constraint(model, sum(qMatrix[sPrime,a,sPrime]*f[[sPrime,a]]/M for a in feasActionSpace) + sum(qMatrix[sa[1],sa[2],sPrime]*f[sa]/M for sa in feasStateActionSpace) == 0)
        if goalTime < time()
            return "TimeOut", [], []
        end
    end
    

    #probabilities sum to one
    @constraint(model, sum(f[sa] for sa in stateActionSpace) == M)

    if speak
        println("Constraints Loaded")
    end

    opCosts = []
    reliabilities = []

    policy = Dict()
    policySeq = Dict()
    for p in ps
        t = goalTime - time()
        if  t < 0
            return "TimeOut", opCosts, reliabilities
        else
            set_time_limit_sec(model, t)
        end

        @objective(model,
        Min,
        sum((cMatrix[sa][1]/M + p*cMatrix[sa][2]/M)*f[sa] for sa in stateActionSpace))


        optimize!(model)

        y = all_variables(model)
        y_solution = value.(y)
        set_start_value.(y, y_solution)

        opCost = 0.0
        reliability = 0.0
        if actionType == "full"
            policy = policyFromFreqs(value.(f), stateSpace)
            opCost = sum(cMatrix[sa][1]*value(f[sa])/M for sa in stateActionSpace)
            reliability = log(sum((cMatrix[sa][2])*value(f[sa])/M for sa in stateActionSpace)) 
        else
            policy = policyFromFreqs(value.(f), stateSpace; actionType = actionType)
            policySeq = policySequencer(policy)
            status, opCost, reliability = mdpPELP(probParams, stateSpace, D, policySeq)
        end

        push!(opCosts, opCost)
        push!(reliabilities, reliability)
    end

    return "Success", opCosts, reliabilities, policy
end

function mdpNonDesignLP_Dual_MultiP(probParams::problemParams, D, ps; speak = false, careful = true, timeLimit = Inf, memLim = 12, actionType = "full", M = 1.0)
    goalTime = Inf
    if timeLimit < Inf
        goalTime = time() + timeLimit
    end

    #Get component parameters
    (;N, alpha, tau, c, p, r) = probParams
    thisProbParams = copy(probParams)
    thisProbParams.p = 1.0

    #start LP model and set attributes
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    if careful
        set_optimizer_attribute(model, "IntegralityFocus", 1)
        set_optimizer_attribute(model, "NumericFocus", 3)
        set_optimizer_attribute(model, "Quad", 1)
        set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
        set_optimizer_attribute(model, "OptimalityTol", 1e-9)
        set_optimizer_attribute(model, "MarkowitzTol", 0.999)
    end

    set_optimizer_attribute(model, "SoftMemLimit", memLim)
    #find maximum state space and state-action space, ignoring null action in worst state
    stateSpace = enumerateStatesESSA(D)
    # stateActionSpace = []
    # for s in stateSpace
    #     actionSpace = []
    #     if actionType == "full"
    #         actionSpace = enumerateFeasibleActionsESSA(s)
    #     elseif actionType == "las1"
    #         actionSpace = enumerateFeasibleActionsESSA_LAS1(s)
    #     elseif actionType == "las2"
    #         actionSpace = enumerateFeasibleActionsESSA_LAS2(s)
    #     else
    #         throw(DomainError(actionType, "Invalid actionType"))
    #     end
    #     for a in actionSpace
    #         #if sum(s[i][2] for i in 1:N) < sum(upper) || a != fill(0, N)
    #         push!(stateActionSpace, [s, a])
    #         #end
    #     end
    # end

    if goalTime < time()
        return "TimeOut", [], []
    end

    if speak
        println("State Space Constructed")
        print(length(stateActionSpace))
        println(" variables")
    end


    #define state-action frequency and binary design variables 
    @variable(model, h[stateSpace] >= 0, start = 0.0)
    @variable(model, g)
    #@variable(model, x[indices] >= 0, start = 0.0) #linear relaxation

    if goalTime < time()
        return "TimeOut", [], []
    end

    #construct matrix of infinitesimal generator and instant costs
    qMatrix = q2(probParams, D; goalTime = goalTime, actionType = actionType)

    if speak
        println("q loaded")
    end 

    if goalTime < time()
        return "TimeOut", [], []
    end

    cMatrix = Dict()
    for sa in stateActionSpace
        cMatrix[sa] = costRateESSA(sa[1],sa[2],thisProbParams, D)
    end

    if goalTime < time()
        return "TimeOut", [], []
    end

    if speak
        println("c loaded")
    end

    @objective(model, Max, g)

    if speak
        println("Constraints Loaded")
    end


    for p in ps
        t = goalTime - time()
        if  t < 0
            return "TimeOut", opCosts, reliabilities
        else
            set_time_limit_sec(model, t)
        end

        for s in stateSpace
            actionSpace = enumerateFeasibleActionsESSA(s)
            for a in actionSpace
                @constraint(model, g - sum(q[s,a,sPrime]*h[sPrime] for sPrime in neighbourhood(postActionState(s,a))) <= cMatrix[[s,a]])
            end
        end

        optimize!(model)
        
        #tbc
        #idea, take reduced costs as frequencies to recalculate opCost and reliability

        # opCost = 0.0
        # reliability = 0.0
        # if actionType == "full"
        #     policy = policyFromFreqs(value.(f), stateSpace)
        #     opCost = sum(cMatrix[sa][1]*value(f[sa])/M for sa in stateActionSpace)
        #     reliability = log(sum((cMatrix[sa][2])*value(f[sa])/M for sa in stateActionSpace)) 
        # else
        #     policy = policyFromFreqs(value.(f), stateSpace; actionType = actionType)
        #     policySeq = policySequencer(policy)
        #     status, opCost, reliability = mdpPELP(probParams, stateSpace, D, policySeq)
        # end

        # push!(opCosts, opCost)
        # push!(reliabilities, reliability)
    end

    #outputs will likely be changed
    return "Success", opCosts, reliabilities, policy
end

function mdpPELP(probParams::problemParams, stateSpace, D, policy; speak = false, careful = true, timeLimit = Inf, memLim = 12)
    goalTime = Inf
    if timeLimit < Inf
        goalTime = time() + timeLimit
    end

    #Get component parameters
    (;N, alpha, tau, c, p, r) = probParams
    thisProbParams = copy(probParams)
    thisProbParams.p = 1.0
    p = 1.0

    #start LP model and set attributes
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    if careful
        set_optimizer_attribute(model, "IntegralityFocus", 1)
        set_optimizer_attribute(model, "NumericFocus", 3)
        set_optimizer_attribute(model, "Quad", 1)
        set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
        set_optimizer_attribute(model, "OptimalityTol", 1e-9)
        set_optimizer_attribute(model, "MarkowitzTol", 0.999)
    end

    set_optimizer_attribute(model, "SoftMemLimit", memLim)


    #define state-action frequency and binary design variables 
    @variable(model, f[stateSpace] >= 0, start = 0.0)

    if goalTime < time()
        return "TimeOut", [], []
    end

    #construct matrix of infinitesimal generator and instant costs
    qMatrix = qWithPolicy(probParams, stateSpace, D, policy)

    if goalTime < time()
        return "TimeOut", [], []
    end

    cMatrix = Dict()
    for s in stateSpace
        cMatrix[s] = costRateESSA(s,policy[s],thisProbParams, D)
    end

    if goalTime < time()
        return "TimeOut", [], []
    end

    if speak
        println("q and c loaded")
    end

    for sPrime in stateSpace
        @constraint(model, sum(qMatrix[s,sPrime]*f[s] for s in stateSpace) == 0) 
    end
    
    if goalTime < time()
        return "TimeOut", [], []
    end

    #probabilities sum to one
    @constraint(model, sum(f[s] for s in stateSpace) == 1.0)

    if speak
        println("Constraints Loaded")
    end

    t = goalTime - time()
    if  t < 0
        return "TimeOut", opCosts, reliabilities
    else
        set_time_limit_sec(model, t)
    end

    optimize!(model)

    opCost = sum(cMatrix[s][1]*value(f[s]) for s in stateSpace)
    reliability = log(sum((cMatrix[s][2])*value(f[s]) for s in stateSpace)) 

    return "Success", opCost, reliability
end

function dcpCostRate(probParams, x)
    (;N,alpha,tau,c,r) = probParams
    qs = alpha ./ (alpha .+ tau)
    if length(x) == 1
        return sum(r[i]*qs[i]*x[i] for i in 1:N) + c[1]*(1 - qs[1]^x[1]) 
    end

    return sum(r[i]*qs[i]*x[i] for i in 1:N) + c[1]*(1 - qs[1]^x[1]) + sum( c[i]*prod(qs[j]^x[j] for j in 1:(i - 1))*(1 - qs[i]^x[i]) for i in 2:N)
end


#input list of unique designs (copies should already be filtered)
function eliminateDominatedDesigns(designs)
    nonDom = []
    for d in designs
        dom = false
        for dComp in designs
            if prod((dComp .- d) .>= 0) && d != dComp
                dom = true
                break
            end
        end
        if !dom
            push!(nonDom, d)
        end
    end

    return nonDom
end

function maskProblemParams(probParams, design)
    (; N, alpha, beta, tau, c, p, r) = probParams

    alphaNew = []
    tauNew = []
    cNew = []
    rNew = []

    for i in 1:N
        if design[i] >= 1
            push!(alphaNew,alpha[i])
            push!(tauNew, tau[i])
            push!(cNew, c[i])
            push!(rNew, r[i])
        end
    end

    return problemParams(length(alphaNew), beta, alphaNew, tauNew, cNew, rNew, p)
end

function mdpDesignHeuristic(probParams::problemParams, C, B, w, W; method = "full-lp", epsilonMin = Inf, epsilonStep = 1.0, pStep = 0.5, speak = false)
    COMP_THRESHOLD = 0.02
    
    #find maximum reliability
    minFailRes = dcpIntMinFailureConstrainedCost(probParams, C, B, w = w, W = W)
    minFailProb = minFailRes[2]
    println(value.(minFailRes[3]))
    #save values
    designs = [value.(minFailRes[3])]
    objVals = [dcpCostRate(probParams, designs[1])]
    logFailRates = [minFailProb]

    #for range of target failure-rates
    targetLFR = epsilonStep
    if epsilonMin < Inf
        targetLFR = epsilonMin 
    end

    while targetLFR < abs(minFailProb)
        #optimise for cost
        res = dcpIntConstrainedFailure(probParams, probLim = -targetLFR, C = C, B = B, w = w, W = W)
        thisDesign = dcpIntBinToVar(res[4])
        
        #if solution is new, save it
        match = false
        for d in designs
            if d == thisDesign
                match = true
                break
            end
        end

        if !match
            push!(designs, thisDesign)
            push!(objVals, res[2])
            push!(logFailRates, res[3])
        end

        targetLFR = -res[3] + epsilonStep
    end              

    #eliminate dominated designs
    dynamicDesigns = eliminateDominatedDesigns(designs)
    dynamicObjVals = []
    dynamicLogFailRates = []
    count = 1
    maxP = 0.0
    for design in dynamicDesigns
        if speak
            println(design)
        end

        probParamsD = maskProblemParams(probParams, design)
        designMasked = []
        
        for d in design
            if d >= 1
                push!(designMasked, d)
            end
        end

        highLogFailRate = 0.0
        faLogFailRate = 0.0
        for k in 1:length(designs)
            if design == designs[k]
                faLogFailRate = logFailRates[k]
                break
            end
        end

        objValsD = []
        logFailRatesD = []
        probParamsD.p = 1.0
        pScale = 10.0^(pStep)
        #println("Min LFR: "*string(faLogFailRate))

        while abs(highLogFailRate- faLogFailRate)/abs(faLogFailRate) >= COMP_THRESHOLD
            if speak
                println(round(abs(highLogFailRate- faLogFailRate)/abs(faLogFailRate), digits = 2))
            end

            probParamsD.p = probParamsD.p*pScale
            if maxP < probParamsD.p 
                maxP = probParamsD.p 
            end

            if !occursin("-lp",method)
                epsilon = min(probParamsD.p*exp(faLogFailRate)/100.0, 0.00001)
                test = rviESSA(probParamsD, designMasked, epsilon, nMax = 10000, delScale = 1, printProgress = false, modCounter = 1000, actionType = method)
                if method != "full"
                    test = rpiESSA(probParamsD, designMasked, test[2], epsilon)
                end 

                push!(objValsD,test[1][1])
                
                push!(logFailRatesD, log(test[1][2]/probParamsD.p))
                highLogFailRate = log(test[1][2]/probParamsD.p)
            else
                actionType = method[1:length(method)-3]
                test = mdpNonDesignLP_MultiP(probParamsD, designMasked, [probParamsD.p]; actionType = actionType)
                push!(objValsD, test[2][1])
                push!(logFailRatesD, test[3][1])
                highLogFailRate = test[3][1]
            end            
        end
        push!(dynamicObjVals, objValsD)
        push!(dynamicLogFailRates, logFailRatesD)
        count = count + 1
    end    

    allObjVals = deepcopy(objVals)
    allLFRs = deepcopy(logFailRates)
    allDesigns = deepcopy(designs)
    for i in 1:length(dynamicObjVals)
        append!(allObjVals, dynamicObjVals[i])
        append!(allLFRs, dynamicLogFailRates[i])
        append!(allDesigns, fill(dynamicDesigns[i], length(dynamicObjVals[i])))
    end
    
    nonDomOJs = []
    nonDomLFRs = []
    nonDomDesigns = []
    for i in 1:length(allObjVals)
        dom = false
        for j in 1:length(allObjVals)
            if i != j && ((allObjVals[i] >= allObjVals[j] && allLFRs[i] > allLFRs[j]) || (allObjVals[i] > allObjVals[j] && allLFRs[i] >= allLFRs[j]))
                dom = true
                break
            end
        end
        if !dom
            push!(nonDomOJs, allObjVals[i])
            push!(nonDomLFRs, allLFRs[i])
            push!(nonDomDesigns, allDesigns[i])
        end
    end

    return designs, objVals, logFailRates, dynamicDesigns, dynamicObjVals, dynamicLogFailRates, nonDomOJs, nonDomLFRs, maxP, nonDomDesigns
end 

function boDop(probParams::problemParams, C, B, w, W; epsilonMin = Inf, epsilonStep = 1.0, speak = false)
    COMP_THRESHOLD = 0.02
    
    #find maximum reliability
    minFailRes = dcpIntMinFailureConstrainedCost(probParams, C, B, w = w, W = W)
    minFailProb = minFailRes[2]
    println(value.(minFailRes[3]))
    #save values
    designs = [value.(minFailRes[3])]
    objVals = [dcpCostRate(probParams, designs[1])]
    logFailRates = [minFailProb]

    #for range of target failure-rates
    targetLFR = epsilonStep
    if epsilonMin < Inf
        targetLFR = epsilonMin 
    end

    while targetLFR < abs(minFailProb)
        #optimise for cost
        res = dcpIntConstrainedFailure(probParams, probLim = -targetLFR, C = C, B = B, w = w, W = W)
        thisDesign = dcpIntBinToVar(res[4])
        
        #if solution is new, save it
        match = false
        for d in designs
            if d == thisDesign
                match = true
                break
            end
        end

        if !match
            push!(designs, thisDesign)
            push!(objVals, res[2])
            push!(logFailRates, res[3])
        end

        targetLFR = -res[3] + epsilonStep
    end              

    return designs, objVals, logFailRates
end 