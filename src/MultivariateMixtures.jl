module MultivariateMixtures

using LinearAlgebra
using Statistics: mean, var, cov, covm
using StatsBase: fit, ConvergenceException
using MultivariateStats: PPCA, FactorAnalysis, outdim
using Distributions
using Distributions: log2π
using Clustering: kmeans, assignments

import Base: show, length, convert
import Distributions: probs, ncomponents, components, component, logpdf, logpdf!
import Statistics: mean

export fit_mm, FinateMixtureModel

include("types.jl")
include("utils.jl")
include("gmm.jl")
include("mfa.jl")

# move to MultivariateStats
length(fa::FactorAnalysis) = outdim(fa)

# auxiliary functions
convert(::Type{MvNormal}, fa::FactorAnalysis) = MvNormal(mean(fa), cov(fa))
logpdf(fa::FactorAnalysis{T}, x::AbstractVector{T}) where {T<:Real} =
    logpdf(convert(MvNormal, fa), x)
logpdf(fa::FactorAnalysis{T}, X::AbstractMatrix{T}) where {T<:Real} =
    logpdf!(Vector{T}(undef, size(X,2)), fa, X)
logpdf!(r::AbstractArray{T}, fa::FactorAnalysis{T}, X::AbstractMatrix{T}) where {T<:Real} =
    logpdf!(r, convert(MvNormal, fa), X)
predict(classifier::Union{MixtureModel,FinateMixtureModel}, data::AbstractVector) =
    findmax(componentwise_logpdf(classifier, data))[end]
predict(classifier::Union{MixtureModel,FinateMixtureModel}, data::AbstractMatrix; dims=2) =
    map(ci->ci[2], findmax(componentwise_logpdf(classifier, (dims == 1 ? data' : data)), dims=2)[end])[:]

end
