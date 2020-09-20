using FastGaussQuadrature, LinearAlgebra

function radauII(n)
    x, w = gaussradau(n)
    x̂ = reverse((1 .- x) / 2)
    ŵ = reverse(w / 2)
    return x̂, ŵ
end

# idea: use block methods for Kernel case!
function volterra(K, G, g, mesh, n)
    cj, wj = radauII(n)
    is = ntuple(identity, length(mesh) - 1)
    results = foldl(is, init=()) do res, i
        pts = @. mesh[i] + cj * (mesh[i+1] - mesh[i])
        acc = g.(pts)
        for j in 1:(i-1)
            pts0 = @. mesh[j] + cj * (mesh[j+1] - mesh[j])
            wts0 = wj * (mesh[i+1] - mesh[i])
            mat = K.(pts', pts0) .* wts0
            acc += mat * res[j]
        end
        return (res..., G.(acc))
    end
    return last(results)
end

## Example

function K(t, s)
    t > s .+ 0.1
end

mesh = 0:0.1:1
G(x) = tanh(x)
g(x) = x
n = 4

volterra(K, G, g, mesh, n)

using BenchmarkTools
@benchmark volterra($K, $G, $g, $mesh, $n)
using ProfileView
ProfileView.@profview for i in 1:100
    volterra(K, G, g, mesh, n)
end