using MultivariateMixtures
using MultivariateStats
using Distributions
using LinearAlgebra
using Test
using Random

@testset "MultivariateMixtures" begin
    include("utils.jl")
    include("gmm.jl")
    include("mfa.jl")
    # include("mppca.jl")
end
