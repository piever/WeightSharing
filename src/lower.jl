using ChainRulesCore, ForwardDiff

function clean!(weights, rgs)
    N = length(rgs)
    for i0 in 1:N, i1 in i0:N
        rg0, rg1 = rgs[i0], rgs[i1]
        weights[rg0, rg1] .= 0
    end
    return weights
end

clean(weights, rgs) = clean!(copy(weights), rgs)

apply(f, x) = f(x)

function update!(z, σ, x, rg)
    z[rg, :] .= apply.(σ, x[rg, :])
    return
end

function update!(z, σ::AbstractArray{<:Number}, x, rg)
    z[rg, :] .= σ[rg, :] .* x[rg, :]
    return
end

function compute_xz(σ, weights, bias, rgs)
    x, z = copy(bias), copy(bias)
    for rg in rgs
        update!(z, σ, x, rg)
        mul!(x, weights[:, rg], z[rg, :], 1, 1)
    end
    return x, z
end

function compute_z(σ, weights, bias, rgs)
    _, z = compute_xz(σ, clean(weights, rgs), bias, rgs)
    return z
end

function ChainRulesCore.rrule(::typeof(compute_z), σ, weights, bias, rgs)
    wts = clean(weights, rgs)
    x, z = compute_xz(σ, wts, bias, rgs)
    function pullback_z(z̄)
        J = ForwardDiff.derivative.(σ, x)
        _, res = compute_xz(J, permutedims(wts), z̄, reverse(rgs))
        NO_FIELDS, DoesNotExist(), clean(res * z', rgs), res, DoesNotExist()
    end
    return z, pullback_z
end

##

using FiniteDiff: finite_difference_gradient

weights = rand(30, 30)
bias = rand(30, 100)
rgs = [10i+1:10i+10 for i in 0:2]

x, z = compute_xz(tanh, weights, bias, rgs)
val, back = rrule(compute_z, tanh, weights, bias, rgs)

z̄ = rand(30, 100)
_, _, a, b, _ = back(z̄)

l = finite_difference_gradient(weights) do weights
    dot(z̄, compute_z(tanh, weights, bias, rgs))
end

l - a

rrule_test(compute_z, z̄, (tanh, nothing), (weights, randn(size(weights)...)), (bias, randn(size(bias)...)), (rgs, nothing))
