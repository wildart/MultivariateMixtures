const Component = Union{PPCA, FactorAnalysis}

struct FinateMixtureModel{VS<:ValueSupport, D<:Distribution, C<:Component, CT<:Real} <: MultivariateMixture{VS,D}
    components::Vector{C}
    prior::Categorical{CT}

    function FinateMixtureModel{VS, D}(cs::Vector{C}, pri::Categorical{CT}) where {VS, C, D, CT}
        length(cs) == ncategories(pri) ||
            error("The number of components does not match the length of prior.")
        new{VS,D,C,CT}(cs, pri)
    end
end

FinateMixtureModel(components::Vector{C}, p::Vector{CT}) where {C<:Component, CT<:Real} =
    FinateMixtureModel{Continuous, MvNormal}(components, Categorical(p))

function show(io::IO, d::FinateMixtureModel)
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

# AbstractMixtureModel Interface

component_type(d::FinateMixtureModel) = typeof(first(d.components))
ncomponents(d::FinateMixtureModel) = length(d.components)
components(d::FinateMixtureModel) = d.components
component(d::FinateMixtureModel, k::Int) = d.components[k]
probs(d::FinateMixtureModel) = probs(d.prior)

function mean(d::FinateMixtureModel)
    K = ncomponents(d)
    π = probs(d)
    m = zeros(length(d))
    for i = 1:K
        πᵢ = π[i]
        if πᵢ > 0.0
            c = component(d, i)
            BLAS.axpy!(πᵢ, mean(c), m)
        end
    end
    return m
end
