@testset "GMM" begin
    Random.seed!(28374544);

    # generate dataset
    d = 2
    theta = [0,3,5,7] * (pi/4)
    k = length(theta)
    X = hcat([MultivariateMixtures.clusters(t, 250, d) for t in theta]...)

    G1 = fit_mle(MvNormal, X[:,1:250])
    GMM = fit_mm(FullNormal, X, k, tol=1e-5)
    @test Distributions.component_type(GMM) <: FullNormal
    @test ncomponents(GMM) == k
    @test probs(GMM) ≈ fill(1/4, k) atol=1e-2
    G2 = component(GMM,2)
    @test mean(G1) ≈ mean(G2) atol=1e-2
    @test cov(G1) ≈ cov(G2) atol=1e-2
    @test length(GMM) == d
    @test size(GMM) == (d,)
    @test mean(GMM) ≈ mean(X, dims=2) atol=1e-2
    @test cov(GMM) ≈ cov(X, dims=2) atol=1e-1

    GMM = fit_mm(FullNormal, X, k, tol=1e-5, homoscedastic=true)
    G3 = component(GMM,4)
    @test mean(G1) ≈ mean(G3) atol=1e-2
    @test cov(G1) ≈ cov(G3) atol=0.5
    @test cov(G3) == cov(component(GMM,1))

    GMM = fit_mm(DiagNormal, X, k, tol=1e-4)
    @test Distributions.component_type(GMM) <: DiagNormal

    GMM = fit_mm(IsoNormal, X, k, tol=1e-4)
    @test Distributions.component_type(GMM) <: IsoNormal

    X = hcat([MultivariateMixtures.clusters(t, 250, 5) for t in theta]...)
    GMM = fit_mm(FullNormal, X, k, tol=1e-3)
    @test length(GMM) == 5

end
