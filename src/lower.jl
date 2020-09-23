using ChainRulesCore, ForwardDiff

function update!(z, σ, x, rg)
    z[rg, :] .= σ.(x[rg, :])
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
    _, z = compute_xz(σ, weights, bias, rgs)
    return z
end

function ChainRulesCore.rrule(::typeof(compute_z), σ, weights, bias, rgs)
    x, z = compute_xz(σ, weights, bias, rgs)
    function pullback_z(z̄)
        wt = permutedims(weights)
        J = ForwardDiff.derivative.(σ, x)
        rgst = reverse(rgs)
        res = compute_z(J, wt, z̄, rgt)
        NO_FIELDS, DoesNotExist(), res * z', res * bias', DoesNotExist()
    end
    return z, pullback_z
end

compute_xz(tanh, rand(30, 30), rand(30, 100), [10i+1:10i+10 for i in 0:2])

using FiniteDiff: finite_difference_gradient

