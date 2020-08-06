@testset "MFA" begin
    Random.seed!(28374544);

    # generate dataset
    theta = [0,3,5,7] * (pi/4)
    m = length(theta)
    X = hcat([clusters(t, 250) for t in theta]...)

    FA1 = fit(FactorAnalysis, X[:,1:250])
    MFA = fit_mm(FactorAnalysis, X, m=m, tol=1e-5)
    FA2 = MultivariateMixtures.component(MFA,3)
    @test mean(FA1) ≈ mean(FA2) atol=1e-2
    @test cov(FA1) ≈ cov(FA2) atol=1e-2

    X = hcat([clusters(t, 250, 10) for t in theta]...)
    FA1 = fit(FactorAnalysis, X[:,1:250], method=:em)
    MFA = fit_mm(FactorAnalysis, X, m=m, tol=1e-5)
    FA2 = MultivariateMixtures.component(MFA,1)
    @test mean(FA1) ≈ mean(FA2) atol=0.01
    @test cov(FA1) ≈ cov(FA2) atol=0.1
end
