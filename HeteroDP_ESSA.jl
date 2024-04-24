println("Start")
using Distributed
addprocs(1)

@everywhere println("Start")

@everywhere begin
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

    ########################################################
    #DYNAMIC STUFF##########################################
    ########################################################

    #cartesian product and repeated cartesian product
    function cartProd(as, bs)
        out = []
        for a in as
            for b in bs
                push!(out, [a,b])
            end
        end

        return out
    end

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
        return problemParams(N, beta, alpha, tau, c, r, p)
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
            nHood = neighbourhood(sPost)
            for sPrime in stateSpace
                q[s,a,sPrime] = 0.0
            end

            total = 0.0
            for i in 1:N
                sRepSucc = copy(sPost)
                sRepFail = copy(sPost)
                sDeg = copy(sPost)
                sRepSucc[i] = sRepSucc[i] .+ [-1,0]
                sRepFail[i] = sRepFail[i] .+ [-1,1]
                sDeg[i] = sDeg[i] .+ [0,1]

                q[s,a,sRepSucc] = tau[i]*sPost[i][1]
                q[s,a,sRepFail] = alpha[i]*sPost[i][1]
                q[s,a,sDeg] = (D[i] - sum(sPost[i]))*alpha[i]

                total += (tau[i] + alpha[i])*sPost[i][1] + (D[i] - sum(sPost[i]))*alpha[i]
            end

            q[s,a,s] = -total
        end
    end 
    return q
end

probParams = problemParams(2, 1.0, [0.1,0.2], [1.0, 1.0], [2.0,1.0], [100.0, 50.0], 1000.0)

function mdpDesignLP(probParams::problemParams, C, B, w, W; minProb = 0.0, speak = false)
    #Get component parameters
    (;N, alpha, tau, c, p, r) = probParams

    #Find maximum mumber of each component according to constraints
    upper = max.(min.(floor.(B ./ C), floor.(W ./ w)), 1)

    #start LP model and set attributes
    model = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "IntegralityFocus", 1)
    set_optimizer_attribute(model, "NumericFocus", 3)
    set_optimizer_attribute(model, "Quad", 1)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    set_optimizer_attribute(model, "OptimalityTol", 1e-9)
    set_optimizer_attribute(model, "MarkowitzTol", 0.999)

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
    qMatrix = q(probParams, upper)
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
        @constraint(model, sum(qMatrix[sa[1],sa[2],sPrime]*f[sa] for sa in stateActionSpace) == 0)
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

    @objective(model,
    Min,
    sum(sum(cMatrix[sa])*f[sa] for sa in stateActionSpace))

    optimize!(model)

    opCost = sum(cMatrix[sa][1]*value(f[sa]) for sa in stateActionSpace)
    reliability = sum(cMatrix[sa][2]*value(f[sa]) for sa in stateActionSpace)/p
    return model, opCost, reliability , f, x
end

function dcpCostRate(probParams, x)
    (;N,alpha,tau,c,r) = probParams
    qs = alpha ./ (alpha .+ tau)
    return sum(r[i]*qs[i]*x[i] for i in 1:N) + c[1]*(1 - qs[1]^x[1]) + sum( c[i]*prod(qs[j]^x[j] for j in 1:(i - 1))*(1 - qs[i]^x[i]) for i in 2:N)
end

Gurobi._is_feasible
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


@everywhere oldName = "./Documents/GitHub/PhD-Code/dcpIntExp1-10.dat"
@everywhere results = deserialize(oldName)
print(keys(results))

for i in 1:1000
    test = [j for j in results["dcps"][i][21] if j > 0]
    if length(test) > 2
        print(i)
        break
    end
end

test = [j for j in results["dcps"][65][21] if j > 0]

[j for j in 1:10 if results["dcps"][65][21][j] > 0]

@everywhere begin
    alpha = results["alphas"][65][8:10]
    beta = 1.0
    tau = results["taus"][65][8:10]
    c = results["cs"][65][8:10]
    r = results["rs"][65][8:10]
end 

js = SharedArray{Float64}(20)
@distributed for i = 1:20
    js[i] = 7 + 0.1*(i - 1)
end

obj1 = SharedArray{Float64}(20)
obj2 = SharedArray{Float64}(20)
@sync @distributed for i = 1:20
    j = js[i]
    p = 10.0^j
    probParams = problemParams(; N = 3, alpha = alpha, beta = beta, tau = tau, c = c, p = p, r = r) 
    D = [1,4,2]
    epsilon = p*exp(results["logFailProbs"][65][21])/100
    test = rviESSA(probParams, D, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "full")
    println(js[i])
    println(test[1][1])
    obj1[i] = test[1][1]
    println(log(test[1][2]/p))
    obj2[i] = log(test[1][2]/p)
    println()
end

StatsPlots.plot(results["objVals"][65][15:21], results["logFailProbs"][65][15:21], seriestype=:scatter, label = "Static Policies")
StatsPlots.plot!(obj1, obj2, seriestype=:scatter, label = "Dynamic Policies (RVIA with FAS)")
xlabel!("Operational Cost")
ylabel!("log-failure-rate")

obj1LAS1 = SharedArray{Float64}(20)
obj2LAS1 = SharedArray{Float64}(20)

@sync @distributed for i = 1:20
    j = js[i]
    p = 10.0^j
    probParams = problemParams(; N = 3, alpha = alpha, beta = beta, tau = tau, c = c, p = p, r = r) 
    D = [1,4,2]
    epsilon = p*exp(results["logFailProbs"][65][21])/100
    test = rviESSA(probParams, D, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "las1")
    println(js[i])
    println(test[1][1])
    obj1LAS1[i] = test[1][1]
    println(log(test[1][2]/p))
    obj2LAS1[i] = log(test[1][2]/p)
    println()
end

StatsPlots.plot!(obj1LAS1, obj2LAS1, seriestype=:scatter, label = "Dynamic Policies (RVIA with LAS1)")

obj1LAS2 = SharedArray{Float64}(20)
obj2LAS2 = SharedArray{Float64}(20)
@sync @distributed for i = 1:20
    j = js[i]
    p = 10.0^j
    probParams = problemParams(; N = 3, alpha = alpha, beta = beta, tau = tau, c = c, p = p, r = r) 
    D = [1,4,2]
    epsilon = p*exp(results["logFailProbs"][65][21])/100
    test = rviESSA(probParams, D, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "las2")
    println(js[i])
    println(test[1][1])
    obj1LAS2[i] = test[1][1]
    println(log(test[1][2]/p))
    obj2LAS2[i] = log(test[1][2]/p)
    println()
end

StatsPlots.plot!(obj1LAS2, obj2LAS2, seriestype=:scatter, label = "Dynamic Policies (RVIA with LAS2)")

obj1LAS1Seq = SharedArray{Float64}(20)
obj2LAS1Seq = SharedArray{Float64}(20)
@sync @distributed for i = 1:20
    j = js[i]
    p = 10.0^j
    probParams = problemParams(; N = 3, alpha = alpha, beta = beta, tau = tau, c = c, p = p, r = r) 
    D = [1,4,2]
    epsilon = p*exp(results["logFailProbs"][65][21])/100
    test = rviESSA(probParams, D, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "las1")
    test = rpiESSA(probParams, D, test[2], epsilon)
    println(js[i])
    println(test[1][1])
    obj1LAS1Seq[i] = test[1][1]
    println(log(test[1][2]/p))
    obj2LAS1Seq[i] = log(test[1][2]/p)
    println()
end

StatsPlots.plot!(obj1LAS1Seq, obj2LAS1Seq, seriestype=:scatter, label = "Dynamic Policies (RVIA with LAS1, Sequenced)")

obj1LAS2Seq = SharedArray{Float64}(20)
obj2LAS2Seq = SharedArray{Float64}(20)
@sync @distributed for i = 1:20
    j = js[i]
    p = 10.0^j
    probParams = problemParams(; N = 3, alpha = alpha, beta = beta, tau = tau, c = c, p = p, r = r) 
    D = [1,4,2]
    epsilon = p*exp(results["logFailProbs"][65][21])/100
    test = rviESSA(probParams, D, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "las2")
    test = rpiESSA(probParams, D, test[2], epsilon)
    println(js[i])
    println(test[1][1])
    obj1LAS2Seq[i] = test[1][1]
    println(log(test[1][2]/p))
    obj2LAS2Seq[i] = log(test[1][2]/p)
    println()
end
StatsPlots.plot!(obj1LAS2Seq, obj2LAS2Seq, seriestype=:scatter, label = "Dynamic Policies (RVIA with LAS2, Sequenced)")

#pi policy testing
obj3 = SharedArray{Float64}(20)
obj4 = SharedArray{Float64}(20)
@everywhere D = [1,4,2]
@everywhere policy = fullyActiveESSA(D)
@everywhere probParams = problemParams(; N = 3, alpha = alpha, beta = beta, tau = tau, c = c, p = 1.0, r = r) 
@everywhere epsilon = exp(results["logFailProbs"][65][21])/100
@sync @everywhere hFA = rpeESSA(probParams, D, policy, epsilon, printProgress = true)[2]
@everywhere stateSpace = enumerateStatesESSA(D)
@sync @distributed for i = 1:20
    j = js[i]
    p = 10.0^j
    hAdjusted = Dict()
    for s in stateSpace
        hAdjusted[s] = [1.0, p] .* hFA[s]
    end

    probParams = problemParams(; N = 3, alpha = alpha, beta = beta, tau = tau, c = c, p = p, r = r) 
    epsilon = p*exp(results["logFailProbs"][65][21])/100
    test = rpiESSA(probParams, D, hAdjusted, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "las1")
    println(js[i])
    println(test[1][1])
    obj3[i] = test[1][1]
    println(log(test[1][2]/p))
    obj4[i] = log(test[1][2]/p)
    println()
end

StatsPlots.plot!(obj3, obj4, seriestype=:scatter, label = "Dynamic Policies (One-Step PI with LAS)")

obj3Full = SharedArray{Float64}(20)
obj4Full = SharedArray{Float64}(20)

@sync @distributed for i = 1:20
    j = js[i]
    p = 10.0^j
    hAdjusted = Dict()
    for s in stateSpace
        hAdjusted[s] = [1.0, p] .* hFA[s]
    end

    probParams = problemParams(; N = 3, alpha = alpha, beta = beta, tau = tau, c = c, p = p, r = r) 
    epsilon = p*exp(results["logFailProbs"][65][21])/100
    test = rpiESSA(probParams, D, hAdjusted, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "full")
    println(js[i])
    println(test[1][1])
    obj3Full[i] = test[1][1]
    println(log(test[1][2]/p))
    obj4Full[i] = log(test[1][2]/p)
    println()
end

StatsPlots.plot!(obj3Full, obj4Full, seriestype=:scatter, label = "Dynamic Policies (One-Step PI with FAS)")

#link sets from literature
taus = [fill(1.0,4),
        fill(1.0,3),
        fill(1.0,4),
        fill(1.0,3),
        fill(1.0,3),
        fill(1.0,4),
        fill(1.0,3),
        fill(1.0,3),
        fill(1.0,4),
        fill(1.0,3),
        fill(1.0,3),
        fill(1.0,4),
        fill(1.0,3),
        fill(1.0,4)]

cs = copy(taus)

rs = [fill(100.0,4),
    fill(100.0,3),
    fill(100.0,4),
    fill(100.0,3),
    fill(100.0,3),
    fill(100.0,4),
    fill(100.0,3),
    fill(100.0,3),
    fill(100.0,4),
    fill(100.0,3),
    fill(100.0,3),
    fill(100.0,4),
    fill(100.0,3),
    fill(100.0,4)]

ps = [[0.9,0.93,0.91,0.95],
    [0.95,0.94,0.93],
    [0.85,0.9,0.87,0.92],
    [0.93,0.87,0.85],
    [0.94,0.93,0.95],
    [0.99,0.98,0.97,0.96],
    [0.91,0.92,0.94],
    [0.81,0.90,0.91],
    [0.97,0.99,0.96,0.91],
    [0.83,0.85,0.9],
    [0.94,0.95,0.96],
    [0.79,0.82,0.85,0.9],
    [0.98,0.99,0.97],
    [0.9,0.92,0.95,0.99]]
alphas = [1.0 ./ ps[i] .- 1.0 for i in 1:14]
Cs = [[1,1,2,2],
    [2,1,1],
    [2,3,1,4],
    [3,4,5],
    [2,2,3],
    [3,3,2,2],
    [4,4,5],
    [3,5,6],
    [2,3,4,3],
    [4,4,5],
    [3,4,5],
    [2,3,4,5],
    [2,3,2],
    [4,4,5,6]]
ws = [[3,4,2,5],
    [8,10,9],
    [7,5,6,4],
    [5,6,4],
    [4,3,5],
    [5,4,5,4],
    [7,8,9],
    [4,7,6],
    [8,9,7,8],
    [6,5,6],
    [5,6,6],
    [4,5,6,7],
    [5,5,6],
    [6,7,6,9]]
probParamses = []
Bs = []
Ws = []

for i in 1:14
    probParams = problemParams(N=length(alphas[i]), alpha = alphas[i], tau = taus[i], c = cs[i], r = rs[i], p = 1.0)
    push!(probParamses, probParams)

    BsI = []
    WsI = []


    res = dcpIntMinMixCostWeightConstrainedFailure(probParams, Cs[i], ws[i], 0.5, -15.0)
    push!(BsI, res[2])
    push!(WsI, res[3])

    push!(BsI, res[2]-1)
    push!(WsI, res[3]+1)

    push!(BsI, res[2]-2)
    push!(WsI, res[3]+2)

    push!(BsI, res[2]+1)
    push!(WsI, res[3]-1)

    push!(BsI, res[2]+2)
    push!(WsI, res[3]-1)



    push!(Bs, BsI)
    push!(Ws, WsI)
end


probParams = problemParams(N=4, beta = 1.0, alpha = alphas[1], tau = taus[1], c = cs[1], r = rs[1], p = 1.0)

res = dcpIntMinCostConstrainedFailure(probParams, ws[1], -14)
res[3]
value.(res[3])

res = dcpIntConstrainedFailure(probParams; probLim = -7.0, M = 1.0, C = Cs[1], B = 4, w = ws[1], W = 11, epsilon = 1.0e-1, timeLimit = 600.0)
dcpIntBinToVar(res[4])
res = dcpIntMinFailureConstrainedCost(probParams, Cs[1], 6, w = ws[1], W = 13)

designs = []
objVals = []
logFailProbs = []
for i in 1:10
    res = dcpIntConstrainedFailure(probParams; probLim = -i, M = 1.0, C = Cs[1], B = 6, w = ws[1], W = 13)
    push!(designs, dcpIntBinToVar(res[4]))
    push!(objVals, res[2])
    push!(logFailProbs, res[3])
end

designs
uniqueDesigns = []
for d in designs
    match = false
    for u in uniqueDesigns
        if d == u
            match = true
            break
        end
    end
    if !match
        push!(uniqueDesigns, d)
    end
end
uniqueDesigns

StatsPlots.plot(objVals, logFailProbs, seriestype=:scatter, label = "Designs", markersize = 7)
xlabel!("Operational Cost")
ylabel!("log-failure-rate")

StatsPlots.savefig("../fairleyl/Documents/GitHub/PhD-Code/PFront1.pdf")

obj1 = []
obj2 = []
js = [i/2 for i in 2:10]
for i in 1:length(js)
    p = 10.0^js[i]
    probParams = problemParams(; N = 2, alpha = alphas[1][2:3], beta = 1.0, tau = taus[1][2:3], c = cs[1][2:3], p = p, r = rs[1][2:3]) 
    D = [2,2]
    epsilon = p*exp(-10)/100
    test = rviESSA(probParams, D, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "las2")
    test = rpiESSA(probParams, D, test[2], epsilon)
    #println(js[i])
    println(test[1][1])
    push!(obj1,test[1][1])
    
    println(log(test[1][2]/p))
    push!(obj2, log(test[1][2]/p))
    println()
end

StatsPlots.plot!(obj1[2:length(js)], obj2[2:length(js)], seriestype=:scatter, label = "Dynamic (Design 1)")
StatsPlots.savefig("../fairleyl/Documents/GitHub/PhD-Code/PFront2.pdf")

obj1_2 = []
obj2_2 = []
#js = [i/3 for i in 2:15]
for i in 1:length(js)
    p = 10.0^js[i]
    probParams = problemParams(; N = 2, alpha = alphas[1][3:4], beta = 1.0, tau = taus[1][3:4], c = cs[1][3:4], p = p, r = rs[1][3:4]) 
    D = [1,2]
    epsilon = p*exp(-10)/100
    test = rviESSA(probParams, D, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "las2")
    test = rpiESSA(probParams, D, test[2], epsilon)
    #println(js[i])
    println(test[1][1])
    push!(obj1_2,test[1][1])
    
    println(log(test[1][2]/p))
    push!(obj2_2, log(test[1][2]/p))
    println()
end

StatsPlots.plot!(obj1_2, obj2_2, seriestype=:scatter, label = "Dynamic (Design 2)")
StatsPlots.savefig("../fairleyl/Documents/GitHub/PhD-Code/PFront3.pdf")

#for each link set i 
    #for constraints j 
        # maximise reliability
        # obtain designs using epsilon constraints up -1 ,..., ciel(min log fail rate)
        # eliminate copied designs
        # plot so far and save
        # eliminate dominated solutions
        # for each remaining design solution 
            #for p = 10 increasing in orders of magnitude until log-fail small enough
                #generate DP solution and save to list of solutions
            # add to plot and save 
        # eliminate dominated solutions and re-plot


staticDesignDict = Dict()
staticObjValDict = Dict()
staticLFRDict = Dict()
dynamicDesignDict = Dict()
dynamicObjValDict = Dict()
dynamicLFRDict = Dict()

#for every component set
for i in 1:14
    println("Problem Set: "*string(i))
    probParams = probParamses[i]
    println(probParams)
    C = Cs[i]
    w = ws[i]

    #for each constraint pair
    for j in 1:5
        println("Constraint Set: "*string(j))
        B = Bs[i][j]
        W = Ws[i][j]

        #find maximum reliability
        minFailRes = dcpIntMinFailureConstrainedCost(probParams, C, B, w = w, W = W)
        minFailProb = minFailRes[2]

        #save values
        designs = [value.(minFailRes[3])]
        objVals = [dcpCostRate(probParams, designs[1])]
        logFailRates = [minFailProb]

        #for range of target failure-rates
        for k in 2:floor(abs(minFailProb))
            #optimise for cost
            res = dcpIntConstrainedFailure(probParams, probLim = -k, C = C, B = B, w = w, W = W)
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
        end

        staticDesignDict[[i,j]] = designs
        staticObjValDict[[i,j]] = objVals
        staticLFRDict[[i,j]] = logFailRates

        #plot solutions so far
        StatsPlots.plot(objVals, logFailRates, seriestype=:scatter, label = "Designs", markersize = 7)
        xlabel!("Operational Cost")
        ylabel!("log-failure-rate")

        title = "../fairleyl/Documents/GitHub/PhD-Code/Pareto-fronts/Set" * string(i) * "Constraints" * string(j) 
        StatsPlots.savefig(title * "static.pdf")


        #eliminate dominated designs
        dynamicDesigns = eliminateDominatedDesigns(designs)
        dynamicObjVals = []
        dynamicLogFailRates = []

        println("Number of Dynamic Designs: "*string(length(dynamicDesigns)))
        count = 1
        for design in dynamicDesigns
            println("Design: "*string(design))
            
            probParamsD = maskProblemParams(probParams, design)
            #println(probParamsD)
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
            pScale = 10.0
            println("Min LFR: "*string(faLogFailRate))

            while abs((exp(highLogFailRate) - exp(faLogFailRate))/exp(faLogFailRate)) >= 0.01
                probParamsD.p = probParamsD.p*pScale
                println("p: "*string(probParamsD.p))
                epsilon = min(probParamsD.p*exp(faLogFailRate)/100.0, 0.00001)
                test = rviESSA(probParamsD, designMasked, epsilon, nMax = 10000, delScale = 1, printProgress = true, modCounter = 1000, actionType = "las2")
                println("RVI Complete")
                test = rpiESSA(probParamsD, designMasked, test[2], epsilon)
                println("PI Complete")

                #println(js[i])
                println("OpCost: "*string(test[1][1]))
                push!(objValsD,test[1][1])
                
                println("LFR: "*string(log(test[1][2]/probParamsD.p)))
                push!(logFailRatesD, log(test[1][2]/probParamsD.p))
                highLogFailRate = log(test[1][2]/probParamsD.p)
                println()
            end

            push!(dynamicObjVals, objValsD)
            push!(dynamicLogFailRates, logFailRatesD)

            StatsPlots.plot!(objValsD, logFailRatesD, seriestype=:scatter, label = "Dynamic (Design "*string(count)*")")
            StatsPlots.savefig(title * "_" * string(count) * ".pdf")

            count = count + 1
        end

        dynamicDesignDict[[i,j]] = dynamicDesigns
        dynamicObjValDict[[i,j]] = dynamicObjVals
        dynamicLFRDict[[i,j]] = dynamicLogFailRates

        out = Dict()
        out["sDesign"] = staticDesignDict
        out["sObj"] = staticObjValDict
        out["sLFR"] = staticLFRDict
        out["dDesign"] = dynamicDesignDict
        out["dObj"] = dynamicObjValDict
        out["dLFR"] = dynamicLFRDict

        f = serialize("../fairleyl/Documents/GitHub/PhD-Code/17AprExp.dat", out)
    end
end

dynamicDesignDict[[5,1]]
pair = [3,1]
StatsPlots.plot(staticObjValDict[pair], staticLFRDict[pair], seriestype=:scatter, label = "Designs", markersize = 7)
xlabel!("Operational Cost")
ylabel!("log-failure-rate")
StatsPlots.savefig("../fairleyl/Documents/GitHub/PhD-Code/nicePlot_" * string(pair) * "0.pdf")

for i in 1:length(dynamicObjValDict[pair])
    markersize = fill(4, length(dynamicObjValDict[pair][i]))
    markersize[length(markersize)] = 7
    StatsPlots.plot!(dynamicObjValDict[pair][i],dynamicLFRDict[pair][i], seriestype=:scatter, label = "Dynamic (Design " * string(i) * ")", markersize = markersize)
    StatsPlots.savefig("../fairleyl/Documents/GitHub/PhD-Code/nicePlot_" * string(pair) * string(i) * ".pdf")
end

allObjVals = deepcopy(staticObjValDict[pair])
allLFRs = deepcopy(staticLFRDict[pair])
for i in 1:length(dynamicObjValDict[pair])
    append!(allObjVals, dynamicObjValDict[pair][i])
    append!(allLFRs, dynamicLFRDict[pair][i])
end

nonDomOJs = []
nonDomLFRs = []
for i in 1:length(allObjVals)
    dom = false
    for j in 1:length(allObjVals)
        if i != j && allObjVals[i] >= allObjVals[j] && allLFRs[i] >= allLFRs[j]
            dom = true
            break
        end
    end
    if !dom
        push!(nonDomOJs, allObjVals[i])
        push!(nonDomLFRs, allLFRs[i])
    end
end

p = sortperm(nonDomOJs)
StatsPlots.plot!(nonDomOJs[p], nonDomLFRs[p], colour = :red, label = "Pareto-front")
StatsPlots.savefig("../fairleyl/Documents/GitHub/PhD-Code/nicePlot" * string(pair) * "_front.pdf")