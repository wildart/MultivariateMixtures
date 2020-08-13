abstract type AbstractEMState{T<:AbstractFloat} end

struct HeteroscedasticGMMState{T<:AbstractFloat} <: AbstractEMState{T}
    πₖ::AbstractArray{T,1}
    μₖ::AbstractArray{T,2}
    Σₖ::AbstractArray{T,3}
end

struct HomoscedasticGMMState{T<:AbstractFloat} <: AbstractEMState{T}
    πₖ::AbstractArray{T,1}
    μₖ::AbstractArray{T,2}
    Σₖ::AbstractArray{T,2}
end

"Gaussian Mixture Model"
function fit_mm(::Type{MV}, X::AbstractMatrix{T}, k::Int = 2;
                tol::Real=1.0e-6,      # convergence tolerance
                maxiter::Integer=1000, # number of iterations
                μs::Union{AbstractArray{T,2}, Nothing} = nothing,
                Σs::Union{AbstractArray{T,3}, Nothing} = nothing
            ) where {T<:AbstractFloat, MV<:MultivariateNormal}

    d, n = size(X)
    Z = similar(X)
    Rₙₖ = zeros(T, n, k)
    Rₙ = zeros(T, n)
    πₖ = zeros(T, k)
    μₖ = zeros(T, d, k)
    Σₖ = zeros(T, d, d, k)

    # initialize parameters
    mX = mean(X, dims=2)
    cX=cov(X, dims=2)
    sc=det(cX)^(1/d)
    for j in 1:k
        Σₖ[:,:,j] .= cX #.+rand(T,d,d) * sqrt(sc)
        μₖ[:,j] .= vec(sqrt(cX)'*randn(T, d) .+ mX)
        πₖ[j] = one(T)/n
    end

    ℒ′ = Δℒ = 0
    for itr in 1:maxiter

        # E Step: Calculate responsibilities (or posterior probability)
        #   Rₙₖ = E[ωₖ|x] = p(ωₖ=1|x) ∝ πₖ⋅p(x|ωₖ) = πₖ⋅(√(2π)^(-d/2)*|Σₖ⁻¹|)⋅exp(-0.5⋅(x-μₖ)ᵀΣₖ⁻¹(x-μₖ))
        # where ωₖ is the mixture indicator variable, s.t. ωₖ = 1 when the data point was generated by mixture ωₖ
        for j in 1:k
            Σ⁻¹ = inv(@view Σₖ[:,:,j])
            broadcast!(-, Z, X, @view μₖ[:,j])
            posterior!(view(Rₙₖ,:,j), πₖ[j], Σ⁻¹, Z)
        end
        # Rₙₖ[Rₙₖ .< eps(T)] .= eps(T)

        # Calculate log-likelihood
        sum!(Rₙ, Rₙₖ)
        ℒ = sum(log.(Rₙ))
        Δℒ = abs(ℒ′ - ℒ)
        @debug "Likelihood" itr=itr ℒ=ℒ Δℒ=Δℒ
        Δℒ < tol && break
        ℒ′ = ℒ

        Rₙₖ ./= Rₙ       # τₙₖ
        # precompute: πₖ = T1 = ∑ₙ(Rₙₖ)/Rₙ
        sum!(πₖ', Rₙₖ)   # τₖ = T1

        # precompute: E[ωₖz|x] = E[ωₖ|x] E[z|ωₖ,x] = μₖ = T2 = ∑ₙ(Rₙₖ⋅x)
        for j in 1:k
            sum!(view(μₖ,:,j), Rₙₖ[:,j]'.*X)
        end

        # precompute: E[ωₖzzᵀ|x] = E[ωₖ|x] E[zzᵀ|ωₖ,x] = Σₖ = T3 = ∑ₙ Rₙₖ⋅x⋅xᵀ
        for i in 1:n
            x = X[:,i]
            for j in 1:k
                view(Σₖ,:,:,j) .+= Rₙₖ[i,j]*x*x'
            end
        end

        # M Step: Calculate parameters
        for j in 1:k
            μⱼ = view(μₖ,:,j)
            Σⱼ = view(Σₖ,:,:,j)
            # Σₖ = (∑ₙ Rₙₖ(x-μₖ)(x-μₖ)ᵀ)/Rₖ = (T3-T2⋅T2'/T1)/T1
            Σⱼ .-= μⱼ*μⱼ'/πₖ[j]
            Σⱼ ./= πₖ[j]
            # μₖ = ∑ₙ(Rₙₖ⋅x)/Rₙ = T2/T1
            μⱼ ./= πₖ[j]
            # πₖ = (∑ₙ Rₙₖ)/n = ∑ₙ (Rₙₖ/Rₙ)
            πₖ[j] /= n
        end
    end

    if Δℒ > tol
        @warn "No convergence" Δℒ=Δℒ tol=tol
    end

    return MixtureModel([MvNormal(μₖ[:,j], Symmetric(Σₖ[:,:,j])) for j in 1:k], πₖ)
end
