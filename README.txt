This repo has three main files:
1) IDDMP.jl
2) functions.jl
3) experiments.jl 
---------------------------------------------------------------------------------------------
IDDMP.jl is the continuously developed and altered file containing both the main 
functions and tests and experiments. It was intended for personal use only, but is
included for completeness.
---------------------------------------------------------------------------------------------
functions.jl seperates out all functions from IDDMP.jl into one file, so that
experiments can be kept seperate in experiments.jl. Nevertheless, it still includes
a lot of deprecated functionality (e.g. early heuristics, dynamic programming algorithms 
and helper functions thereof). Some specific functions of interest for those who want to 
experiment are as follows:

function problemParams(; N = 0, beta = 1.0, alpha = [], tau = [], c = [], r = [], p = 0.0)

This function creates a problemParams struct containing problem parameters for the component
set. N counts the number of component types, beta is deprecated and should be left as 1,
alpha, tau, c, and r are as described in the paper. p is penalty for system failure


function dcpIntConstrainedFailure(probParams::problemParams; probLim = 0.0, M = 1.0, C = 0.0, B = Inf, w = 0.0, W = Inf, epsilon = 1.0e-1, timeLimit = 600.0)

This function solves epsilon-delta-DOP-PC. The variable names don't quite match the paper. probParams
is a problemParams struct. probLim is the epsilon (i.e. target LFR value). M can be ignored. 
C is an installation cost vector, and B is a budget. w is a weight vector, W is maximum weight.
epsilon is actually delta. timeLimit is a time limit in seconds for Gurobi, however this would only
be needed for extreme problem instances.


function dcpIntMinFailureConstrainedCost(probParams::problemParams, C, B; w = 0.0, W = Inf)

This function is F-DOP. Inputs are consistent with the above.


function mdpDesignLP_MultiP(probParams::problemParams, C, B, w, W, ps; speak = false, careful = true, timeLimit = Inf, memLim = 12, actionType = "full")

This function solves the MILP formulation of p-IDDMP directly for a range of p values "ps". careful is 
a switch for certain Gurobi attributes. memLim is a memory limit in GB. actionType should be 
left alone, as it is a holdover from previous work 


function mdpNonDesignLP_MultiP(probParams::problemParams, D, ps; speak = false, careful = true, timeLimit = Inf, memLim = 12, actionType = "full", M = 1.0)

This function solves the LP formulation of p-DMP for a range of p values "ps". "speak" is a switch
for extra outputs. M is not needed.


function mdpDesignHeuristic(probParams::problemParams, C, B, w, W; method = "full-lp", epsilonMin = Inf, epsilonStep = 1.0, pStep = 0.5, speak = false)

This function applies APP to BO-IDDMP. method = "full-lp" reflects the method used in the paper, 
other options are deprecated. epsilonMin is the starting value of epsilon, epsilonStep is the 
increment value. pStep is the exponent of Delta-p, and it is assumed that p_min = Delta_p.
---------------------------------------------------------------------------------------------
experiments.jl are the experiments that are used in the paper. Notably, the first experiment is old,
but its outputs were used for illustrative purposes, and thus it is included for completions sake.

