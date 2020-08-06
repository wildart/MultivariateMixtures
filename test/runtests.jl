using MultivariateMixtures
using MultivariateStats
using Test
using Random

# generate clusters
function clusters(t, n, d=2)
    ct = cos(t)
    st = sin(t)

    x0 = randn(n) .+ 5.0
    y0 = randn(n) .* 0.3

    x = x0 .* ct .- y0 .* st
    y = x0 .* st .+ y0 .* ct
    if d <= 2
        [x y]'
    else
        vcat([x y]', rand(d-2, n))
    end
end


@testset "MultivariateMixtures" begin
    include("mfa.jl")
    # include("mppca.jl")
end
