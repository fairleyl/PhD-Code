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

numRuns = 100
N = 1000
taus = []
rels = []
alphas = []
cs = []
Cs = []
rs = []
TAU_MAX = 1.0
C0_MAX = 1.0
R_MAX = 100.0
maxCost = 10
Random.seed!(12345)
for i in 1:numRuns
    tau = rand(N).*TAU_MAX 
    push!(taus, tau)

    rel = 0.9 .+ 0.1*rand(N)
    push!(rels, rel)

    alpha = tau .* ((1 ./ rel) .- 1)
    push!(alphas, alpha)

    c = rand(N).*C0_MAX
    c = sort(c)
    push!(cs, c)

    r = 1 .+ rand(N).*(R_MAX - 1)
    push!(rs, r)

    C = sample([i for i in 1:maxCost], N)
    push!(Cs, C)
end

named = "./Documents/GitHub/PhD-Code/dcpIntExp1-1000.dat"
results = deserialize(named)
taus = results["taus"]
rels = results["rels"]
alphas = results["alphas"]
cs = results["cs"]
Cs = results["Cs"]
rs = results["rs"]
beta = 1.0
p = 0.0
epsilon = 1.0e-1
for N in [1000]
    
    #iterate over each random link set
    for i in 87:numRuns
        dcpsI = []
        objValsI = []
        logFailProbsI = []
        minCostsI = []
        termStatI = []
        print("Run: ")
        println(i)

        #print("rel: ")
        #println(round.(rels[i], digits = 5))

        #print("c: ")
        #println(round.(cs[i], digits = 5))

        #print("r: ")
        #println(round.(rs[i], digits = 5))

        probParams = problemParams(N = N, alpha = alphas[i], beta = beta, tau = taus[i], c = cs[i], p = p, r = rs[i])
        
        #solve for different log-fail constraints, down to a roughly 1/billion chance
        #print("probLim: ")
        for probLim in -1:-1:-21    
            #print("Max log-failure probability: ")
            print(probLim)     
            print(", ")
            res = dcpIntConstrainedFailure(probParams; probLim = probLim, M = 1.0, C = 0.0, B = Inf, epsilon = epsilon, timeLimit = 60.0)
            resCost = dcpIntMinCostConstrainedFailure(probParams, Cs[i], probLim)

            model = res[1]
            objVal = res[2]
            logFailProb = res[3]
            x = res[4]
            termStat = termination_status(res[1])
            minCost = resCost[2]

            #println(solution_summary(model))
            #print("Usage Costs: ")
            #println(round(objVal, digits = 5))
            push!(objValsI, objVal)

            #print("Log Failure Probability: ")
            #println(round(logFailProb, digits = 5))
            push!(logFailProbsI, logFailProb)

            push!(dcpsI, dcpIntBinToVar(x))
            push!(minCostsI, minCost)
            push!(termStatI, termStat)
        end
        println()
        push!(results["dcps"], dcpsI)
        push!(results["objVals"], objValsI)
        push!(results["logFailProbs"], logFailProbsI)
        push!(results["minCosts"], minCostsI)
        push!(results["termStats"], termStatI)
        f = serialize(named, results)
    end
end