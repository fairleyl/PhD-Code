println("Start") #Julia Warm-up
include("functions.jl")

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

cs = deepcopy(taus)

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

#The Bs and Ws are different constraint values for an older experiment not included in paper, except for the APP illustration
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

###################################################################################
#This code is an early deprecated version of APP that uses dynamic programming instead of LPs
#The output of this code was used to generate the figures used to illustrate APP.
###################################################################################
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

        f = serialize("../fairleyl/Documents/GitHub/PhD-Code/1JulyExp.dat", out)
    end
end

#Choosing different pairs produces plots for different "component set - constraint" combinations. [1,5] is used in the paper
pair = [1,5]
StatsPlots.plot([0],[0], markersize = 0, primary = false, thickness_scaling = 1.3)
StatsPlots.plot!(staticObjValDict[pair], staticLFRDict[pair], seriestype=:scatter, label = "DOP", markersize = 7)
xlabel!("Operational Cost")
ylabel!("LFR")
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


###########################################################
#Experiment comparing exact method to APP, and DOP alone 
###########################################################
numReps = 10 #number of repetitions used to obtain average solve time
outDict = Dict()
#f = serialize("Exp30Apr24.big", outDict) #uncomment if this file doesn't already exist
outDict = deserialize("Exp30Apr24.big")

#APP method
for i in 1:14
    println("Set: "*string(i))
    for j in 3:5
        println("Constraint "*string(j))
        probParams = copy(probParamses[i])
        C = Cs[i]
        w = ws[i]
        B = minimum(w)*j
        W = B

        ####
        #FAS
        ####
        print("FAS: ")
        outDict[i,j,"FAS-LP","time"] = []
        for k in 1:numReps
            t = time()
            outDict[i,j,"FAS-LP","out"] = mdpDesignHeuristic(probParams, C, B, w, W; method = "full-lp")
            heuristicTime = time() - t 
            push!(outDict[i,j,"FAS-LP","time"], heuristicTime)
            print(string(k)*", ")
        end
        println() 
        f = serialize("Exp30Apr24.big", outDict)

    end
end

#Exact method
for i in 1:14
    println("Set: "*string(i))
    for j in 3:5
        println("Constraint "*string(j))
        probParams = copy(probParamses[i])
        C = Cs[i]
        w = ws[i]
        B = minimum(w)*j #Choosing constraint values as specified in the paper
        W = B


        maxP = outDict[i,j,"FAS-LP","out"][9]
        print("LP: ")
        outDict[i,j,"LP","time"] = []
        for k in 1:numReps
            LPstartT = time()
            obj1_LP = []
            obj2_LP = []
            js = [i/2 for i in 1:2*ceil(log10(maxP))]
            ps = [10.0^js[i] for i in 1:length(js)]
            outDict[i,j,"LP","out"] = mdpDesignLP_MultiP(probParams, C, B, w, W, ps; speak = false, careful = true, timeLimit = 300)


            LPTime = time() - LPstartT
            push!(outDict[i,j,"LP","time"], LPTime)
            print(string(k)*", ")
            GC.gc()
        end
        println()
        f = serialize("Exp30Apr24.big", outDict)
    end
end

#DOP 
numReps = 10
dopTimes = Dict()
for i in 1:14
    println("Set: "*string(i))
    for j in 3:5
        println("Constraint "*string(j))
        probParams = copy(probParamses[i])
        C = Cs[i]
        w = ws[i]
        B = minimum(w)*j
        W = B

        times = []
        for k in 1:numReps
            t = time()
            out = boDop(probParams, C, B, w, W; epsilonStep = 1.0)
            heuristicTime = time() - t 
            push!(times, heuristicTime)
            print(string(k)*", ")
        end
        dopTimes[i,j] = times
    end
end


#######################
#Result plots for above
#######################
outDict = deserialize("Exp30Apr24.big")
i = 14
j = 5

B = j*minimum(ws[i])
staticObjVals = outDict[i,j,"FAS","out"][2]
staticLFRs = outDict[i,j,"FAS","out"][3]

lpObjVals = outDict[i,j,"LP","out"][2]
lpLFRs = outDict[i,j,"LP","out"][3]


fasLPObjVals = outDict[i,j,"FAS-LP","out"][7]
fasLPLFRs = outDict[i,j,"FAS-LP","out"][8]

StatsPlots.plot(staticObjVals, staticLFRs, seriestype=:scatter, label = "DOP", markersize = 7)
xlabel!("OpCost")
ylabel!("LFR")
StatsPlots.plot!(lpObjVals, lpLFRs, seriestype=:scatter, label = "LP")
StatsPlots.plot!(fasLPObjVals, fasLPLFRs, seriestype=:scatter, label = "FAS-LP")

mean(outDict[i,j,"LP","time"])
mean(outDict[i,j,"FAS-LP","time"])

outDict[1,5,"FAS-LP","out"]
for i in 1:14
    println("Problem Set "*string(i))
    for j in 3:5
        numSol = length(outDict[i,j,"FAS-LP","out"][1])
        println("Budget "*string(j)*", Solutions: "*string(numSol))
    end
end


#####################
#Varying usage costs 
#####################
#i and j control problem instance
i = 6
j = 5
begin
    c1 = 1
    c2 = 1
    probParams = copy(probParamses[i])
    (; N, alpha, beta, tau, c, p, r) = probParams
    C = copy(Cs[i])
    w = copy(ws[i])
    B = minimum(w)*j #constraint set following description in paper
    W = B

    new_c = [c1,c2,1,1]
    p = sortperm(new_c)
    probParams = problemParams(4, beta, alpha[p], tau[p], new_c[p], r[p], 1.0)
    C = C[p]
    w = w[p]
    test = mdpDesignHeuristic(probParams, C, B, w, W; method = "full-lp", epsilonStep = 0.01, epsilonMin = 1.0)

    
    dynamicObjVals = test[7]
    dynamicLFRs = test[8]

    p2 = sortperm(dynamicObjVals)
    p3 = sortperm(test[2])
    println(test[1][p3])
    println(test[2][p3])
    println(test[3][p3])
    println(p)
    numSols = length(test[1])

    
end
StatsPlots.plot(test[2][p3], test[3][p3], seriestype =:scatter, label = "DOP", markersize = 7)
StatsPlots.plot!(dynamicObjVals[p2], dynamicLFRs[p2], seriestype =:scatter, label = "c = "*string((c1,c2)))

xlabel!("Operational Cost")
ylabel!("LFR")



#######################################################
#Varying repair costs##################################
#######################################################

#Select problem instance
i = 6
j = 5

#Vary r1 and r2 in this code to explore effect
begin
    r1 = 100
    r2 = 100
    probParams = copy(probParamses[i])
    (; N, alpha, beta, tau, c, p, r) = probParams
    C = copy(Cs[i])
    w = copy(ws[i])
    B = minimum(w)*j
    W = B

    new_r = [r1,r2,100,100]

    probParams = problemParams(4, beta, alpha, tau, c, new_r, 1.0)
    test = mdpDesignHeuristic(probParams, C, B, w, W; method = "full-lp", epsilonStep = 0.01, speak = true)

    
    dynamicObjVals = test[7]
    dynamicLFRs = test[8]

    p2 = sortperm(dynamicObjVals)
    p3 = sortperm(test[2])
    println(test[1][p3])
    println(test[2][p3])
    println(test[3][p3])
    println(p)
    numSols = length(test[1])

end
StatsPlots.plot(test[2][p3], test[3][p3], seriestype =:scatter, label = "DOP", markersize = 7)
StatsPlots.plot!(dynamicObjVals[p2], dynamicLFRs[p2], seriestype =:scatter, label = "IDDMP") #, xlim = (20.99999,21.00001), ylim = (-19.7,-18.5)
xlabel!("Operational Cost")
ylabel!("LFR")
#StatsPlots.savefig("../fairleyl/Documents/GitHub/PhD-Code/r="*string(r1)*"-"*string(r2)*".pdf")
test[4]

######################################
#Varying event rates
######################################
i = 6
j = 5

mults = [1.0,5.0,10.0]
for mult1 in mults
    for mult2 in mults
        r1 = 300.0
        r2 = 100.0
        probParams = copy(probParamses[i])
        (; N, alpha, beta, tau, c, p, r) = probParams
        C = copy(Cs[i])
        w = copy(ws[i])
        B = minimum(w)*j
        W = B

        new_r = [r1, r2, 100.0, 100.0]
        probParams = problemParams(4, beta, alpha .* [mult1, mult2, 1.0,1.0], tau .* [mult1, mult2, 1.0,1.0], c, new_r, 1.0)
        test = mdpDesignHeuristic(probParams, C, B, w, W; method = "full-lp", epsilonStep = 0.01, speak = true)

        
        dynamicObjVals = test[7]
        dynamicLFRs = test[8]
        dynamicDesigns = test[10]
        p2 = sortperm(dynamicObjVals)
        p3 = sortperm(test[2])
        println(test[1][p3])
        println(test[2][p3])
        println(test[3][p3])
        println(p)
        numSols = length(test[1])

        StatsPlots.plot(test[2][p3], test[3][p3], seriestype =:scatter, label = "DOP", markersize = 7, thickness_scaling = 1.3)
        StatsPlots.plot!(dynamicObjVals[p2], dynamicLFRs[p2], seriestype =:scatter, label = "IDDMP") #, xlim = (20.99999,21.00001), ylim = (-19.7,-18.5)
        xlabel!("Operational Cost")
        ylabel!("LFR")
        StatsPlots.savefig("../fairleyl/Documents/GitHub/PhD-Code/mult="*string(mult1,mult2)*".pdf")
    end
end