@testset "MFA" begin

    # generate dataset
    theta = [0,3,5,7] * (pi/4)
    K = length(theta)
    X = hcat([clusters(t, 250) for t in theta]...)

    # z = FactorAnalysis(rand(3), rand(3,3), rand(3))
    # MixtureModel([z, z], [0.3, 0.7])
    M = fit_mm(FactorAnalysis, X, k=K)

end
