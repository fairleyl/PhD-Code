print("Start")

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

import Base.copy

ENV["GUROBI_HOME"] = "~/gurobi1001/linux64"
ENV["GRB_LICENSE_FILE"]="/home/fairleyl/gurobi1001/gurobi.lic"

const GRB_ENV = Gurobi.Env()

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

function dcpIntConstrainedFailure(probParams::problemParams; probLim = 0.0, M = 1.0, C = 0.0, B = Inf, epsilon = 1.0e-1, timeLimit = 600.0)
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

    @variable(model, x[1:N], Int)

    @constraint(model, sum(log(qs[i])*x[i] for i in 1:N) <= probLim)

    @objective(model,
    Min,
    sum(C[i]*x[i] for i in 1:N))

    optimize!(model)
    return model, objective_value(model), x 
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

function dcpIntMinFailureConstrainedCost(probParams::problemParams, C, B)
    (;N, alpha, tau, c, p, r) = probParams
    ps = tau ./ (tau + alpha)
    qs = 1 .- ps
    upper = max.(floor.(B ./ C), 1)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "IntegralityFocus", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    #set_optimizer_attribute(model, "Quad", 1)
    #set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    #set_optimizer_attribute(model, "OptimalityTol", 1e-9)
    #set_optimizer_attribute(model, "MarkowitzTol", 0.999)

    @variable(model, x[1:N], Int)

    @constraint(model, sum(C[i]*x[i] for i in 1:N) <= B)

    @objective(model,
    Min,
    sum(log(qs[i])*x[i]) for i in 1:N)

    optimize!(model)

    return model, objective_value(model), x
end

p = 0.0
numRuns = 100
maxCost = 10
epsilon = 1.0e-1
beta = 1.0
newName = "./Documents/GitHub/PhD-Code/dcpIntExp2-"*string(N)*".dat"


for N in [5,10,50,100,500,1000]
    print("N = ")
    println(N)

    #load previous file
    oldName = "./Documents/GitHub/PhD-Code/dcpIntExp1-"*string(N)*".dat"
    oldResults = deserialize(oldName)
    taus = oldResults["taus"]    
    alphas = oldResults["alphas"]
    rels = oldResults["rels"]
    cs = oldResults["cs"]
    Cs = oldResults["Cs"]
    rs = oldResults["rs"]
    minCosts = oldResults["minCosts"]

    #new field for runtimes
    oldResults["time"] = []

    #load constrained file
    newName = "./Documents/GitHub/PhD-Code/dcpIntExp2-"*string(N)*".dat"
    newResults = deserialize(newName)

    #new field for runtimes
    newResults["time"] = []

    #Bs = newResults["Bs"]
    for i in 1:numRuns
        print(i)
        print(", ")
        tau = taus[i]
        alpha = alphas[i]
        c = cs[i]
        r = rs[i]
        C = Cs[i]
        rel = rels[i]

        probParams = problemParams(; N = N, alpha = alpha, beta = beta, tau = tau, c = c, p = p, r = r)
        #objValsI = []
        #dcpsI = []
        #logFailProbsI = []
        #termStatsI = []
        #BsI = []
        timeUncsI = []
        timeConsI = []
        for probLim in -1:-1:-21
            prevCost = sum(C .* oldResults["dcps"][i][-probLim])
            print(probLim)
            print(", ")
            #println(prevCost)
            #println(minCosts[i][-probLim])
            B = (prevCost + minCosts[i][-probLim])/2
            #push!(BsI, B)
            timeUnc = @elapsed dcpIntConstrainedFailure(probParams; probLim = probLim, M = 1.0, C = C, B = Inf, epsilon = epsilon, timeLimit = 60.0)
            timeCon = @elapsed dcpIntConstrainedFailure(probParams; probLim = probLim, M = 1.0, C = C, B = B, epsilon = epsilon, timeLimit = 60.0)
            #print(solution_summary(res[1]))
            #model = res[1]
            #objVal = res[2]
            #logFailProb = res[3]
            #x = res[4]
            #termStat = termination_status(res[1])

            #push!(objValsI, objVal)
            #push!(logFailProbsI, logFailProb)

            #push!(dcpsI, dcpIntBinToVar(x))
            #push!(termStatsI, termStat)
            push!(timeUncsI, timeUnc)
            push!(timeConsI, timeCon)
        end
        println()

        #push!(newResults["dcps"], dcpsI)
        #push!(newResults["objVals"], objValsI)
        #push!(newResults["logFailProbs"], logFailProbsI)
        #push!(newResults["termStats"], termStatsI)
        #push!(newResults["Bs"], BsI)
        push!(oldResults["time"], timeUncsI)
        push!(newResults["time"], timeConsI)
        f = serialize(oldName, oldResults)
        f = serialize(newName, newResults)
    end
    println()
end

N = 5
newName = "./Documents/GitHub/PhD-Code/dcpIntExp2-"*string(N)*".dat"
newResults = deserialize(newName)
dcps = newResults["dcps"]
run = 3
l = [(i,dcps[run][21][i]) for i in 1:N if dcps[run][21][i] > 0]

variety = []
for i in 1:100
    varI = []
    for j in 1:21
        usedComps = [(k,dcps[i][j][k]) for k in 1:N if dcps[i][j][k] > 0]
        push!(varI, length(usedComps))
    end
    push!(variety, varI)
end

variety

plots = []
for N in [5,10,50,100,500,1000]
    newName = "./Documents/GitHub/PhD-Code/dcpIntExp1-"*string(N)*".dat"
    newResults = deserialize(newName)
    time = newResults["time"]N = 500
    newName = "./Documents/GitHub/PhD-Code/dcpIntExp1-"*string(N)*".dat"
    newResults = deserialize(newName)
    dcps = newResults["dcps"]
    run = 3
    l = [(i,dcps[run][21][i]) for i in 1:100 if dcps[run][21][i] > 0]
    
    variety = []
    for i in 1:100
        varI = []
        for j in 1:21
            usedComps = [(k,dcps[i][j][k]) for k in 1:N if dcps[i][j][k] > 0]
            push!(varI, length(usedComps))
        end
        push!(variety, varI)
    end
    
    variety
    p = StatsPlots.plot(time[1])
    for i in 2:100
        StatsPlots.plot!(p, time[i])
    end
    push!(plots, p)
end

varietiesUnc = []
varietiesCon = []
for N in [5,10,50,100,500,1000]
    oldName = "./Documents/GitHub/PhD-Code/dcpIntExp1-"*string(N)*".dat"
    newName = "./Documents/GitHub/PhD-Code/dcpIntExp2-"*string(N)*".dat"
    oldResults = deserialize(oldName)
    newResults = deserialize(newName)
    oldDcps = oldResults["dcps"]
    newDcps = newResults["dcps"]

    varietyUncN = []
    varietyConN = []
    for i in 1:100
        varUncI = []
        varConI = []
        for j in 1:21
            usedCompsUnc = [(k,oldDcps[i][j][k]) for k in 1:N if oldDcps[i][j][k] > 0]
            usedCompsCon = [(k,newDcps[i][j][k]) for k in 1:N if newDcps[i][j][k] > 0]
            push!(varUncI, length(usedCompsUnc))
            push!(varConI, length(usedCompsCon))
        end
        push!(varietyUncN, varUncI)
        push!(varietyConN, varConI)
    end
    push!(varietiesUnc, varietyUncN)
    push!(varietiesCon, varietyConN)
end

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
#Cs = [...]
#Ws = [...]
