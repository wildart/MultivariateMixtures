module MultivariateMixtures

using LinearAlgebra
using Statistics: mean, var, cov, covm
using StatsBase: fit, ConvergenceException
using MultivariateStats: PPCA, FactorAnalysis
using Distributions

import Base: show
import Distributions: probs, ncomponents, components, component
import Statistics: mean

export fit_mm, FinateMixtureModel

include("types.jl")
include("utils.jl")
include("gmm.jl")
include("mfa.jl")

end
