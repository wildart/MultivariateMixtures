using MultivariateMixtures
using MultivariateStats
using Test

# generate clusters
function clusters(t, n)
	ct = cos(t)
	st = sin(t)

	x0 = randn(n) .+ 5.0
	y0 = randn(n) .* 0.3

	x = x0 .* ct .- y0 .* st
	y = x0 .* st .+ y0 .* ct

	[x y]'
end


@testset "MultivariateMixtures" begin
    include("mfa.jl")
    # include("mppca.jl")
end
