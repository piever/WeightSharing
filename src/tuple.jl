using LinearAlgebra

function compute_xz(σ, weights::Tuple, bias)
    x, z = copy(bias), copy(bias)
    i1 = 0
    for weight in weights
        i1 = size(x, 1) - size(weight, 1)
        i0 = i1 - size(weight, 2)
        z[i0+1:i1, :] .= σ.(x[i0+1:i1, :])
        x[i1+1:end, :] .+= weight * z[i0+1:i1, :]
    end
    z[i1+1:end, :] .= σ.(x[i1+1:end, :])
    return x, z
end

weights = (rand(3, 10), rand(5, 5), rand(3, 2))

bias = rand(13, 5)

σ = tanh