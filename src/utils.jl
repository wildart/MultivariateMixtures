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

function posterior!(r, α, Σ⁻¹, Z)
    d = size(Z, 1)
    c = α*(2π)^(-d/2)*sqrt(det(Σ⁻¹))
    colwise_dot!(r, Z, Σ⁻¹ * Z)
    broadcast!(e->c*exp(-e/2), r, r)
end

"Calculate responsibilities (or posterior probability)"
function logpost!(r, α, Σ⁻¹::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T <: AbstractFloat}
    d = size(Z, 1)
    colwise_dot!(r, Z, Σ⁻¹ * Z)
    r ./= -2
    r .+= log(α)+log(2π)*(-d/2)+log(sqrt(det(Σ⁻¹)))
end

function logpost!(r, α, Ch::Cholesky, Z::AbstractMatrix{T}) where {T <: AbstractFloat}
    d = size(Z, 1)
    logdetL⁻¹ = sum(log.(1 ./ diag(Ch.U)))
    tmp = similar(Z)
    ldiv!(tmp, Ch, Z)
    sum!(r', broadcast!(*, tmp, Z, tmp))
    r ./= -2
    r .+= log(α)+log(2π)*(-d/2)+logdetL⁻¹
end

"""
    logsumexp!(logΣexp, w)

Return log(∑exp(w)). Modifies the weight vector to `w = exp(w-offset)`
Uses a numerically stable algorithm with offset to control for overflow and `log1p` to control for underflow.
References:
https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
"""
function logsumexp!(Σ::AbstractVector{T}, w::AbstractMatrix{T};
                    tmpind::AbstractVector{CartesianIndex{2}} = Array{CartesianIndex{2}}(undef, length(Σ)),
                    offset::AbstractVector{T}=similar(Σ)) where {T <: AbstractFloat}
    findmax!(offset, tmpind, w)
    w .= exp.(w .- offset)
    sumexept!(Σ, w, tmpind) # Σ = ∑wₑ-1
    Σ .= log1p.(Σ) .+ offset
end

# function sumexept!(s, w, i)
#     w[i] .-= 1
#     sum!(s, w)
#     w[i] .+= 1
#     s
# end

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

function logreps!(Σ::AbstractVector{T}, w::AbstractMatrix{T}, offset::AbstractVector{T}) where {T <: AbstractFloat}
    n, d = size(w)
    maximum!(offset, w)
    @inbounds @simd for i in 1:n
        o = offset[i]
        wr = view(w, i, :)
        Σ[i] = 0
        for e in wr
            Σ[i] += exp(e - o)
        end
        Σ[i] = log1p(Σ[i]-1) + o
        wr .= exp.(wr .- Σ[i])
    end
end

function logreps!(Σ::AbstractVector{T}, w::AbstractMatrix{T}) where {T <: AbstractFloat}
    Σ .= log.(sum!(exp, Σ, w))
    w .= exp.(w .- Σ)
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
    [A[i] == j ? 1.0 : 0.0 for i in 1:n, j in 1:k]
end

function initialize_random(X::AbstractMatrix{T}, k::Int) where {T <: AbstractFloat}
    d, n = size(X)
    Rₙₖ = rand(T, n, k)
    Rₙₖ ./ sum(Rₙₖ, dims=2)
end

function stats(X::AbstractMatrix{T}, Rₙₖ::AbstractMatrix{T}) where {T<:AbstractFloat}
    πₖ = sum(Rₙₖ, dims=1) .+ 10eps(T)
    πₖ, X*Rₙₖ
end

function stats!(πₖ::AbstractVector{T}, μₖ::AbstractMatrix{T},
                X::AbstractMatrix{T},  Rₙₖ::AbstractMatrix{T}) where {T<:AbstractFloat}
    sum!(πₖ', Rₙₖ)
    πₖ .+= 10eps(T)
    mul!(μₖ,X,Rₙₖ)
end

function cov!(::Type{FullNormal}, πₖ::AbstractVector{T}, μₖ::AbstractMatrix{T}, Σₖ::AbstractArray{T,3},
              X::AbstractMatrix{T}, Rₙₖ::AbstractMatrix{T}) where {T<:AbstractFloat}
    fill!(Σₖ, zero(T))
    d, k = size(μₖ)
    n = sum(πₖ)
    for j in 1:k
        μ = view(μₖ,:,j)
        μ ./= πₖ[j]
        z = X .- μ
        Σ = view(Σₖ, :, :, j)
        Σ .= Rₙₖ[:,j]'.*z*z'/πₖ[j]
        πₖ[j] /= n
    end
    Σₖ
end

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

    d, k = size(X,1), size(Rₙₖ,2)
    πₖ = zeros(T, k)
    μₖ = zeros(T, d, k)
    stats!(πₖ, μₖ, X, Rₙₖ)
    Σₖ = zeros(T, d, d, k)
    cov!(MV, πₖ, μₖ, Σₖ, X, Rₙₖ)
    πₖ, μₖ, Σₖ, Rₙₖ
end

# function cov!(::Type{FullNormal}, πₖ::AbstractVector{T}, μₖ::AbstractMatrix{T}, Σₖ::AbstractArray{T,3},
#               X::AbstractMatrix{T}, Rₙₖ::AbstractMatrix{T}) where {T<:AbstractFloat}
#     fill!(Σₖ, zero(T))
#     d, k = size(μₖ)
#     for i in 1:size(X,2)
#         x = view(X,:,i)
#         xxᵀ = x*x'
#         for j in 1:k
#             view(Σₖ,:,:,j) .+= Rₙₖ[i,j]*xxᵀ
#         end
#     end
#     for j in 1:k
#         μⱼ = view(μₖ,:,j)
#         Σⱼ = view(Σₖ,:,:,j)
#         Σⱼ .-= μⱼ*μⱼ'/πₖ[j]
#         Σⱼ ./= πₖ[j]
#         μⱼ ./= πₖ[j]
#         πₖ[j] /= n
#     end
#     πₖ, μₖ, Σₖ, Rₙₖ
# end
