using MultivariateMixtures: colwise_dot!, logsumexp, logsumexp!, stats!, covariances!

@testset "Utils" begin
    Random.seed!(42)

    a = [1 2 3; 4 5 6.]
    b = [4 5 6; 7 8 9.]
    r = zeros(size(a,2))

    colwise_dot!(r, a, b)
    @testset for i in 1:length(r)
        @test r[i]  == a[:,i] ⋅ b[:,i]
    end

    c = a'/10
    d = copy(c)
    logsumexp!(r, d)
    q = mapslices(r-> log(sum(exp, r)), c, dims=2)
    @testset for (v,w) in  zip(r,q)
        @test v ≈ w atol=1e-15
    end
    @testset for (v,w) in  zip(eachrow(c),eachrow(d))
        @test  exp.(v .- maximum(v)) == w
    end

    s = big(2e-20) - big(1e-40)
    d = [1e-20 log(1e-20)]
    r = zeros(1)
    logsumexp!(r, d)
    @test r[1] ≈ s atol=1e-35
    @test d ≈ [1.0 1e-20] atol=1e-35

    d = [1e-20 log(1e-20)]
    r = logsumexp(d)
    @test r[1] ≈ s atol=1e-35

    r[1] = 0.0
    logsumexp!(r, d, similar(r))
    @test r[1] ≈ s atol=1e-35
    @test d ≈ [1.0 1e-20] atol=1e-35

    d = 2
    s = 100
    k = 3
    n = k*s
    muori = [0.0 0.0 5.0; -5.0 0.0 5.0]
    Cori = [5.0; 1.0; 0.05]
    R = vcat(repeat([1.0 0.0 0.0], s), repeat([0.0 1.0 0.0], s), repeat([0.0 0.0 1.0], s))
    X = vcat(Cori[1]*randn(s,d).+muori[:,1]', Cori[2]*randn(s,d).+muori[:,2]', Cori[3]*randn(s,d).+muori[:,3]')'

    nk, mu = zeros(k), zeros(d,k)
    stats!(nk, mu, X, R)
    @testset for i in nk
        @test i ≈ s
    end
    @testset for i in 1:k
        @test mu[:, i] ≈ muori[:, i] atol=1.0
    end

    C1 = zeros(d,d,k)
    covariances!(C1, nk, mu, X, R, FullNormal, aux=similar(X))
    @testset for i in 1:k
        @test sqrt(det(C1[1,1,i])) ≈ Cori[i] atol=0.3
    end

    C2 = zeros(d,1,k)
    covariances!(C2, nk, mu, X, R, DiagNormal)
    @testset for i in 1:k
        @test sqrt(det(C2[1,1,i])) ≈ Cori[i] atol=0.3
    end

    C3 = zeros(1,1,k)
    covariances!(C3, nk, mu, X, R, IsoNormal, aux=similar(mu))
    @testset for i in 1:k
        @test sqrt(C3[1,1,i]) ≈ Cori[i] atol=0.3
    end

    A,B,C = MultivariateMixtures.new_params(FullNormal, Float64, d, k)
    @test size(A) == (k,)
    @test size(B) == (d,k)
    @test size(C) == (d,d,k)
    A,B,C = MultivariateMixtures.new_params(DiagNormal, Float64, d, k)
    @test size(A) == (k,)
    @test size(B) == (d,k)
    @test size(C) == (d,1,k)
    A,B,C = MultivariateMixtures.new_params(IsoNormal, Float64, d, k)
    @test size(A) == (k,)
    @test size(B) == (d,k)
    @test size(C) == (1,1,k)
    A,B = MultivariateMixtures.auxiliary(FullNormal, X, k)
    @test size(A) == (n,)
    @test size(B) == (d,n)
    A,B = MultivariateMixtures.auxiliary(DiagNormal, X, k)
    @test size(A) == (n,)
    @test size(B) == (d,k)
    A,B = MultivariateMixtures.auxiliary(IsoNormal, X, k)
    @test size(A) == (n,)
    @test size(B) == (d,k)

    F1 = MultivariateMixtures.precision(FullNormal, C1[:,:,1])
    @test F1 isa LowerTriangular
    logpdf!(view(R,:,1), X, mu[:,1], F1)
    r = logpdf(MvNormal(mu[:,1], C1[:,:,1]), X)
    @test sum(view(R,:,1) .- r) ≈ 0.0 atol=1e-10

    F2 = MultivariateMixtures.precision(DiagNormal, C2[:,:,2])
    @test F2 isa Diagonal
    logpdf!(view(R,:,2), X, mu[:,2], F2)
    r = logpdf(MvNormal(mu[:,2], Diagonal(vec(C2[:,:,2]))), X)
    @test sum(view(R,:,2) .- r) ≈ 0.0 atol=1e-10

    F3 = MultivariateMixtures.precision(IsoNormal, C3[:,:,3])
    @test F3 isa UniformScaling
    logpdf!(view(R,:,3), X, mu[:,3], F3)
    r = logpdf(MvNormal(mu[:,3], sqrt(C3[:,:,3][])), X)
    @test sum(view(R,:,3) .- r) ≈ 0.0 atol=1e-10

end
