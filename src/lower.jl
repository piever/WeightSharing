using ChainRulesCore: NO_FIELDS, DoesNotExist
import ChainRulesCore: rrule
using ChainRulesTestUtils: rrule_test
using ForwardDiff: derivative
using Random, LinearAlgebra

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

struct Multiplier{T}
    x::T
end

(m::Multiplier)(val) = m.x * val

function update!(z, σ, x, rg)
    z[rg, :] .= apply.(σ[rg, :], x[rg, :])
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

function rrule(::typeof(compute_z), σ, weights, bias, rgs)
    wts = clean(weights, rgs)
    x, z = compute_xz(σ, wts, bias, rgs)
    function pullback_z(z̄)
        J = Multiplier.(derivative.(σ, x))
        _, res = compute_xz(J, permutedims(wts), z̄, reverse(rgs))
        NO_FIELDS, DoesNotExist(), clean(res * z', rgs), res, DoesNotExist()
    end
    return z, pullback_z
end

##

using FillArrays

weights = rand(30, 30)
bias = rand(30, 100)
rgs = [10i+1:10i+10 for i in 0:2]
σ = Fill(tanh, size(bias))

x, z = compute_xz(σ, weights, bias, rgs)
z̄ = rand(30, 100)

rrule_test(compute_z, z̄, (σ, nothing), (weights, randn(size(weights)...)), (bias, randn(size(bias)...)), (rgs, nothing))
