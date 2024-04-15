println("Start")
using Distributed
addprocs(4)

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
end 

@everywhere function fullyActiveESSA(D)
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

@everywhere function postActionState(s, a)
    N = length(a)
    sPrime = copy(s)
    for i in 1:N
        sPrime[i] = sPrime[i] .+ (a[i] .* [1, -1])
    end
    return sPrime
end

@everywhere function actionSequencer(s, policy)
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

@everywhere function policySequencer(policy)
    newPolicy = Dict()
    stateSpace = keys(policy)
    for s in stateSpace
        newPolicy[s] = actionSequencer(s, policy)
    end

    return newPolicy
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
