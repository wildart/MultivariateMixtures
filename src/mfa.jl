"Mixture of Factor Analyzers"
function fit_mm(::Type{FactorAnalysis}, X::AbstractMatrix{T};
                m::Integer = 2,        # the number of factor analyzers to use
                k::Integer = 1,        # the number of factors in each analyzer
                tol::Real=1.0e-6,      # convergence tolerance
                maxiter::Integer=1000, # number of iterations
                μs::Union{AbstractMatrix{T}, Nothing} = nothing
            ) where T<:Real

    d, n = size(X)
    CMVN = (2π)^(-d/2)
    mX=mean(X, dims=2)
    cX=cov(X, dims=2)
    sc=det(cX)^(1/d)

    # initialize parameters
    πⱼ = fill(one(T)/m, m)
    Wⱼ = zeros(T,d,k,m)
    μⱼ = zeros(T,d,m)
    for j in 1:m
        Wⱼ[:,:,j] = randn(d,k) * sqrt(sc/k)
        μⱼ[:,j] = μs !== nothing ? μs[:,j] : (randn(d)' * sqrt(cX))' + mX # zeros(d)
    end
    Ψⱼ = hcat(fill(diag(cX) .+ eps(), m)...) #fill(0.01, d, m)
    hᵢⱼ = zeros(T,m,n)
    E₍zₗxᵢωⱼ₎ = zeros(T,k,n,m)
    E₍zzᵀₗxᵢωⱼ₎ = zeros(T,k,k,m)

    L_old = 0.0
    for itr in 1:maxiter
        # E Step
        # 1) Calculate hᵢⱼ, E₍zₗxᵢωⱼ₎
        for j in 1:m
            Ψ⁻¹ = diagm(0 => 1 ./ Ψⱼ[:, j])
            W = Wⱼ[:,:,j]
            WᵀΨ⁻¹ = W'*Ψ⁻¹
            # V⁻¹ = inv(I + WᵀΨ⁻¹*W)
            # Σ⁻¹ = Ψ⁻¹ - Ψ⁻¹*W*V⁻¹*WᵀΨ⁻¹
            Σ⁻¹ = Ψ⁻¹ - Ψ⁻¹*W*inv(I + WᵀΨ⁻¹*W)*WᵀΨ⁻¹
            detΣ⁻¹ = sqrt(det(Σ⁻¹))
            # @debug "$j: |Σ⁻¹|" detΣ⁻¹
            Y = X .- view(μⱼ,:,j)
            Σ⁻¹Y = Σ⁻¹*Y
            hᵢⱼ[j, :] = πⱼ[j]*(CMVN*detΣ⁻¹).*exp.(-0.5.*sum(Y.*Σ⁻¹Y, dims=1))
            # Ez[:,:,j] = V⁻¹*WᵀΨ⁻¹*Y # W'*Σ⁻¹*Y
            E₍zₗxᵢωⱼ₎[:,:,j] = W'Σ⁻¹Y
        end

        # 2) Calculate log(L)
        hᵢⱼ[hᵢⱼ .< eps()] .= eps()
        hᵢ = sum(hᵢⱼ, dims=1)
        L = sum(log.(hᵢ))

        # Check exit conditions
        if itr > 1
            chg = L - L_old
            if chg < 0
                @warn "log likelihood decreased" itr=itr change=chg
            end
            @debug "log(L)=$L" itr=itr increment=chg hᵢ=hᵢ Ψⱼ=vec(Ψⱼ)
            if (chg < tol)
                break
            end
        end
        L_old = L

        # 3) Calculate πⱼ
        πᵢⱼ = hᵢⱼ./hᵢ
        πⱼ = sum(πᵢⱼ, dims=2)

        # 4) Calculate E₍zzᵀₗxᵢωⱼ₎
        for j in 1:m
            Ψ⁻¹ = diagm(0 => 1 ./ Ψⱼ[:, j])
            W = Wⱼ[:,:,j]
            WᵀΨ⁻¹ = W'*Ψ⁻¹
            Σ⁻¹ = Ψ⁻¹ - Ψ⁻¹*W*inv(I + WᵀΨ⁻¹*W)*WᵀΨ⁻¹
            Y = X .- view(μⱼ,:,j)
            βⱼ = W'*Σ⁻¹
            Sᵢ = (πᵢⱼ[j, :]'.*Y)*Y'/πⱼ[j]
            E₍zzᵀₗxᵢωⱼ₎[:,:,j] = I - βⱼ*W + βⱼ*Sᵢ*transpose(βⱼ)
        end

        # M Step
        for j in 1:m
            ZIᵀ = hcat(E₍zₗxᵢωⱼ₎[:,:,j]', ones(n))
            QWnum = πᵢⱼ[j, :]'.*X*ZIᵀ
            QWden1 = πⱼ[j]*E₍zzᵀₗxᵢωⱼ₎[:,:,j]
            QWden23 = E₍zₗxᵢωⱼ₎[:,:,j]*πᵢⱼ[j, :]
            QWden = [QWden1 QWden23; QWden23' πⱼ[j]]
            R = QWnum/QWden

            Wⱼ[:,:,j] = R[:,1:end-1]
            μⱼ[:,j] = R[:,end]

            Ψⱼ[:,j] = diag((πᵢⱼ[j, :]'.*X)*X' - R*QWnum')/πⱼ[j]
        end
    end
    return MixtureModel([FactorAnalysis(μⱼ[:,j], Wⱼ[:,:,j], Ψⱼ[:,j]) for j in 1:m], vec(πⱼ/n))
end
