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
