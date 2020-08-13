using MultivariateMixtures
using MultivariateStats
using Distributions
using Test
using Random

@testset "MultivariateMixtures" begin
    include("gmm.jl")
    include("mfa.jl")
    # include("mppca.jl")
end
