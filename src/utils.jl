"
    colwise_dot!(r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)

In-place dot product between columns of matrices `a` and `b`
"
function colwise_dot!(r::AbstractArray, a::AbstractMatrix, b::AbstractMatrix)
    n = length(r)
    @assert n == size(a, 2) == size(b, 2) && size(a, 1) == size(b, 1) "Inconsistent argument dimensions."
    for j = 1:n
        v = zero(promote_type(eltype(a), eltype(b)))
        @simd for i = 1:size(a, 1)
            @inbounds v += a[i, j]*b[i, j]
        end
        r[j] = v
    end
    return r
end

function pdf!(r, X, μ, Σ, Z)
    broadcast!(-, Z, X, μ)
    c = 1/(2π)^(length(μ)/2)*sqrt(det(Σ))
    colwise_dot!(r, Z, inv(Σ) * Z)
    broadcast!(e->c*exp(-e/2), r, r)
end

"Calculate responsibilities (or posterior probability)"
function logpdf!(r, X::AbstractMatrix, μ::AbstractVector, L::LowerTriangular,
                 Z::AbstractMatrix=similar(X))
    broadcast!(-, Z, X, μ)
    MultivariateMixtures.colwise_dot!(r, Z, L \ Z)
    r ./= -2
    r .+= -(log(2π)*length(μ)+ logdet(L))/2
end

function logpdf!(r, X::AbstractMatrix{T}, μ::AbstractVector{T}, D::Diagonal{T},
                 Z::AbstractMatrix{T}=zeros(T,1,1)) where {T<:AbstractFloat}
    m, n = size(X)
    @inbounds @simd for j in 1:n
        s = zero(T)
        for i in 1:m
            s += D[i,i] * abs2(X[i,j] - μ[i])
        end
        r[j] = s
    end
    r ./= -2
    r .+= -(log(2π)*length(μ) + logdet(D))/2
end

function logpdf!(r, X::AbstractMatrix{T}, μ::AbstractVector{T}, D::UniformScaling{T},
                 Z::AbstractMatrix{T}=zeros(T,1,1)) where {T<:AbstractFloat}
    m, n = size(X)
    logdetD = m*log(D[1,1])
    @inbounds @simd for j in 1:n
        s = zero(T)
        for i in 1:m
            s += abs2(X[i,j] - μ[i])
        end
        r[j] = D[1,1] *s
    end
    r ./= -2
    r .+= -(log(2π)*length(μ) + logdetD)/2
end

function logpdf!(r, X::AbstractMatrix, μ::AbstractVector, Σ::Hermitian, Z=similar(X))
    broadcast!(-, Z, X, μ)
    MultivariateMixtures.colwise_dot!(r, Z, inv(Σ) * Z)
    r ./= -2
    r .+= -(log(2π)*length(μ) + logdet(Σ))/2
end

function logpdf!(r, X::AbstractMatrix, μ::AbstractVector, Ch::Factorization, Z=similar(X))
    broadcast!(-, Z, X, μ)
    # define aux vars
    tmp = similar(Z)
    # r .= Z.*(inv(Ch)*Z)
    ldiv!(tmp, Ch, Z)
    sum!(r', broadcast!(*, tmp, Z, tmp))
    r ./= -2
    r .+= -(log(2π)*length(μ) + logdet(Ch))/2
end


"""
    logsumexp!(logΣexp, w)

In-place row-wise calculation of `log( ∑ exp(rᵢ) )` of row `rᵢ` in the matrix `w`.
Modifies the original weight vector to `w = exp(w-offset)`.
Uses a numerically stable algorithm with offset to control for overflow and `log1p` to control for underflow.

References:
- https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
- https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
"""
function logsumexp!(Σ::AbstractVector{T}, w::AbstractMatrix{T};
                    tmpind = Array{CartesianIndex{2}}(undef, length(Σ)),
                    offset = similar(Σ)) where {T <: AbstractFloat}
    findmax!(offset, tmpind, w)
    w .= exp.(w .- offset)
    sumexept!(Σ, w, tmpind) # Σ = ∑wₑ-1
    Σ .= log1p.(Σ) .+ offset
end

logsumexp(w::AbstractMatrix{T})  where {T <: AbstractFloat} =
    logsumexp!(zeros(T, size(w,1)), copy(w))

function sumexept!(s, w, idx)
    n, d = size(w)
    @inbounds for i in 1:n
        l, m = idx[i][1], idx[i][2]
        s[l] = 0
        w[l, m] -= 1
        @simd for j in 1:d
            s[l] += w[l, j]
        end
        w[l, m] += 1
    end
    s
end

function logsumexp!(Σ::AbstractVector{T}, w::AbstractMatrix{T}, offset::AbstractVector{T}) where {T <: AbstractFloat}
    n, d = size(w)
    maximum!(offset, w)
    @inbounds @simd for i in 1:n
        o = offset[i]
        wr = view(w, i, :)
        Σ[i] = -1
        for e in wr
            Σ[i] += exp(e - o)
        end
        Σ[i] = log1p(Σ[i]) + o
        wr .= exp.(wr .- Σ[i])
    end
end

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

function initialize_kmeans(X::AbstractMatrix{T}, k::Int) where {T <: AbstractFloat}
    d, n = size(X)
    C = kmeans(X, k)
    A = assignments(C)
    [A[i] == j ? one(T) : zero(T) for i in 1:n, j in 1:k]
end

function initialize_random(X::AbstractMatrix{T}, k::Int) where {T <: AbstractFloat}
    d, n = size(X)
    Rₙₖ = rand(T, n, k)
    Rₙₖ ./ sum(Rₙₖ, dims=2)
end

function stats(X::AbstractMatrix{T}, Rₙₖ::AbstractMatrix{T}) where {T<:AbstractFloat}
    nₖ = sum(Rₙₖ, dims=1) .+ 10eps(T)
    nₖ, X*Rₙₖ./nₖ
end

function stats!(nₖ::AbstractVector{T}, μₖ::AbstractMatrix{T},
                X::AbstractMatrix{T},  Rₙₖ::AbstractMatrix{T}) where {T<:AbstractFloat}
    sum!(nₖ', Rₙₖ)
    nₖ .+= 10eps(T)
    mul!(μₖ,X,Rₙₖ)
    μₖ ./= nₖ'
end

function covariances!(Σₖ::AbstractArray{T,3}, nₖ::AbstractVector{T}, μₖ::AbstractMatrix{T},
                      X::AbstractMatrix{T}, Rₙₖ::AbstractMatrix{T}, ::Type{FullNormal};
                      covreg::Real=1e-6, aux::AbstractMatrix{T}=similar(X)) where {T<:AbstractFloat}
    d, k = size(μₖ)
    for j in 1:k
        broadcast!(-, aux, X, @view μₖ[:,j])
        Σ = view(Σₖ, :, :, j)
        Σ .= (Rₙₖ[:,j]'.*aux)*aux'/nₖ[j] .+ covreg
    end
    Σₖ
end

function covariances!(Σₖ::AbstractArray{T,3}, nₖ::AbstractVector{T}, μₖ::AbstractMatrix{T},
                      X::AbstractMatrix{T}, Rₙₖ::AbstractMatrix{T}, ::Type{DiagNormal};
                      covreg::Real=1e-6, aux::AbstractMatrix{T}=similar(μₖ)) where {T<:AbstractFloat}
    d, k = size(μₖ)
    S = reshape(Σₖ, d, k)
    mul!(S, (X.*X), Rₙₖ)
    S ./= nₖ'
    broadcast!(*, aux, μₖ, (X*Rₙₖ))
    aux ./= nₖ'
    S .-= 2*aux - μₖ.^2 .- covreg
    Σₖ
end

function covariances!(Σₖ::AbstractArray{T,3}, nₖ::AbstractVector{T}, μₖ::AbstractMatrix{T},
                      X::AbstractMatrix{T}, Rₙₖ::AbstractMatrix{T}, ::Type{IsoNormal};
                      covreg::Real=1e-6, aux::AbstractMatrix{T}=similar(μₖ)) where {T<:AbstractFloat}
    d, k = size(μₖ)
    Σs = reshape(aux, d, 1, k)
    covariances!(Σs, nₖ, μₖ, X, Rₙₖ, DiagNormal, covreg=covreg)
    Σₖ .= mean(Σs, dims=1)
end

new_params(::Type{FullNormal}, ::Type{T}, d, k) where {T<:AbstractFloat} =
    (zeros(T, k), zeros(T, d, k), zeros(T, d, d, k))

new_params(::Type{DiagNormal}, ::Type{T}, d, k) where {T<:AbstractFloat} =
    (zeros(T, k), zeros(T, d, k), zeros(T, d, 1, k))

new_params(::Type{IsoNormal}, ::Type{T}, d, k) where {T<:AbstractFloat} =
    (zeros(T, k), zeros(T, d, k), zeros(T, 1, 1, k))

auxiliary(::Type{FullNormal}, X::AbstractMatrix{T}, k::Int) where {T<:AbstractFloat} =
    (zeros(T, size(X,2)), similar(X))

auxiliary(::Type{DiagNormal}, X::AbstractMatrix{T}, k::Int) where {T<:AbstractFloat} =
    (zeros(T, size(X,2)), zeros(T, size(X,1), k))

auxiliary(::Type{IsoNormal}, X::AbstractMatrix{T}, k::Int) where {T<:AbstractFloat} =
    auxiliary(DiagNormal, X, k)

"""
    initialize(X::AbstractMatrix{T}, k::Int; init=:kmenas)

Initialization of the Gaussian mixture parameters.
"""
function initialize(::Type{MV}, X::AbstractMatrix{T}, k::Int;
                    init=:kmeans) where {T<:AbstractFloat, MV<:MultivariateNormal}
    Rₙₖ = if init == :kmeans
        initialize_kmeans(X, k)
    elseif init == :random
        initialize_random(X, k)
    else
        error("Unimplemented initialization algorithm: $init")
    end

    d, n = size(X)
    nₖ, μₖ, Σₖ = new_params(MV, T, d, k)
    stats!(nₖ, μₖ, X, Rₙₖ)
    covariances!(Σₖ, nₖ, μₖ, X, Rₙₖ, MV)
    nₖ/n, μₖ, Σₖ, Rₙₖ
end

function factorize(::Type{FullNormal}, Σ::AbstractMatrix)
    Ch = cholesky!(Hermitian(Σ, :L), check=false)
    !issuccess(Ch) && @debug "Cholesky factorization failed" err=Ch.info
    if Ch.info<0
        error("Cholesky factorization failed ($(Ch.info)).\\
              Try to decrease the number of components or increase `covreg`: $covreg.")
    end
    Ch.L
end

function factorize(::Type{DiagNormal}, Σ::AbstractMatrix)
    any(iszero, Σ) && error("Factorization failed. Try to decrease the number of components")
    Diagonal(1 ./ sqrt.(vec(Σ)))
end

function factorize(::Type{IsoNormal}, Σ::AbstractMatrix)
    any(iszero, Σ) && error("Factorization failed. Try to decrease the number of components")
    (1/sqrt(Σ[]))I
end

function repaircov!(Σ)
    F = eigen(Σ)
    V = F.values
    # replace negative eigenvalues by zero
    V .= max.(V, 0)
    # reconstruct covariance matrix
    Σ .= F.vectors * Diagonal(V) * F.vectors'
end

caplog(e::T) where {T<:AbstractFloat} = log(ifelse(e>0, e, floatmin(T)))
