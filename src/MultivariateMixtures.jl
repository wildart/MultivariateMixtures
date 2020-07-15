module MultivariateMixtures

using LinearAlgebra
using Statistics: mean, var, cov, covm
using StatsBase: fit, ConvergenceException
using MultivariateStats: PPCA, FactorAnalysis
using Distributions: Categorical, MvNormal, ncategories, logpdf

import Base: show
import Distributions: probs

export MixtureModel, fit_mm

const Component = Union{PPCA, FactorAnalysis, MvNormal}

struct MixtureModel{C<:Component, CT<:Real}
    components::Vector{C}
    prior::Categorical{CT}

    function MixtureModel(cs::Vector{C}, pri::Categorical{CT}) where {C, CT}
        length(cs) == ncategories(pri) ||
            error("The number of components does not match the length of prior.")
        new{C,CT}(cs, pri)
    end
end

MixtureModel(components::Vector{C}, p::Vector{CT}) where {C<:Component, CT<:Real} =
    MixtureModel(components, Categorical(p))

"""
    component_type(d::MixtureModel)
The type of the components of `d`.
"""
component_type(d::MixtureModel{C,CT}) where {C,CT} = C

ncomponents(d::MixtureModel) = length(d.components)
components(d::MixtureModel) = d.components
component(d::MixtureModel, k::Int) = d.components[k]

probs(d::MixtureModel) = probs(d.prior)

function show(io::IO, d::MixtureModel)
    K = ncomponents(d)
    pr = probs(d)
    println(io, "Mixture of $(component_type(d))(K = $K)")
    Ks = min(K, 8)
    for i = 1:Ks
        print(io, "components[$i] (prior = $(pr[i])): ")
        println(io, component(d, i))
    end
    if Ks < K
        println(io, "The rest are omitted ...")
    end
end

include("mfa.jl")

end
