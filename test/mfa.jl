@testset "MFA" begin
    Random.seed!(28374544);

    # generate dataset
    d = 2
    theta = [0,3,5,7] * (pi/4)
    k = length(theta)
    X = hcat([MultivariateMixtures.clusters(t, 250, d) for t in theta]...)

    FA1 = fit(FactorAnalysis, X[:,1:250])
    MFA = fit_mm(FactorAnalysis, X, k, tol=1e-5)
    @test MultivariateMixtures.component_type(MFA) <: FactorAnalysis
    @test ncomponents(MFA) == k
    @test probs(MFA) ≈ fill(1/4, k) atol=1e-2
    FA2 = component(MFA,3)
    @test indim(FA2) == d
    @test outdim(FA2) == 1
    @test mean(FA1) ≈ mean(FA2) atol=1e-2
    @test cov(FA1) ≈ cov(FA2) atol=1e-2
    @test_broken length(MFA) == d
    @test_broken size(MFA) == (d,)
    @test_broken mean(MFA) ≈ mean(X, dims=2) atol=1e-2
    @test_broken cov(MFA) ≈ cov(X, dims=2) atol=1e-1

    X = hcat([MultivariateMixtures.clusters(t, 250, 10) for t in theta]...)
    FA1 = fit(FactorAnalysis, X[:,1:250], method=:em)
    MFA = fit_mm(FactorAnalysis, X, k, tol=1e-5)
    FA2 = component(MFA,1)
    @test mean(FA1) ≈ mean(FA2) atol=0.01
    @test cov(FA1) ≈ cov(FA2) atol=0.1

    MFA = fit_mm(FactorAnalysis, X, k, factors=3, tol=1e-5, maxiter=10)
    @test outdim(component(MFA,3)) == 3
end
