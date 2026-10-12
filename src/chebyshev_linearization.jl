# chebyshev_linearization.jl - Exact dyadic product linearization and multi-index transforms
# for tensor Chebyshev basis in TSSOS.

"""
    count_degrees(mono::Vector{UInt16}) -> Dict{UInt16, Int}

Counts the exponent/degree of each variable in a multi-index represented as a sorted Vector{UInt16}.
E.g., UInt16[1, 1, 2] -> Dict(1 => 2, 2 => 1).
"""
function count_degrees(mono::Vector{UInt16})
    degs = Dict{UInt16, Int}()
    for v in mono
        degs[v] = get(degs, v, 0) + 1
    end
    return degs
end

"""
    degrees_to_multiindex(degs::Dict{UInt16, Int}) -> Vector{UInt16}

Converts degree dictionary back to a sorted Vector{UInt16}.
Omits variables with degree 0 (since T_0 = 1).
"""
function degrees_to_multiindex(degs::Dict{UInt16, Int})
    res = UInt16[]
    for v in sort!(collect(keys(degs)))
        d = degs[v]
        if d > 0
            for _ in 1:d
                push!(res, v)
            end
        end
    end
    return res
end

"""
    cheby_product(beta::Vector{UInt16}, gamma::Vector{UInt16}) -> Vector{Tuple{Vector{UInt16}, Float64}}

Computes the exact tensor Chebyshev product T_beta(x) * T_gamma(x) on [-1, 1]^n using
the dyadic linearization identity:
    T_p(x) * T_q(x) = 0.5 * (T_{p+q}(x) + T_{|p-q|}(x))  for p, q >= 1.

For K shared active coordinates (where both beta and gamma have positive degree),
produces at most 2^K multi-indices each with dyadic rational coefficient 2^(-K).
If K == 0, returns [(sadd(beta, gamma), 1.0)].
"""
function cheby_product(beta::Vector{UInt16}, gamma::Vector{UInt16})::Vector{Tuple{Vector{UInt16}, Float64}}
    if isempty(beta)
        return [(copy(gamma), 1.0)]
    end
    if isempty(gamma)
        return [(copy(beta), 1.0)]
    end

    deg_b = count_degrees(beta)
    deg_g = count_degrees(gamma)

    common_vars = UInt16[v for v in keys(deg_b) if haskey(deg_g, v)]
    K = length(common_vars)
    if K == 0
        return [(sort!(vcat(beta, gamma)), 1.0)]
    end

    # Pre-collect non-common variables
    base_degs = Dict{UInt16, Int}()
    for (v, d) in deg_b
        if !haskey(deg_g, v)
            base_degs[v] = d
        end
    end
    for (v, d) in deg_g
        if !haskey(deg_b, v)
            base_degs[v] = d
        end
    end

    p_vec = Int[deg_b[v] for v in common_vars]
    q_vec = Int[deg_g[v] for v in common_vars]
    coeff = 1.0 / (2.0^K)

    # Accumulate terms in a dictionary to combine identical multi-indices
    terms = Dict{Vector{UInt16}, Float64}()
    total_combinations = 1 << K

    for mask in 0:(total_combinations - 1)
        cur_degs = copy(base_degs)
        for k in 1:K
            v = common_vars[k]
            p = p_vec[k]
            q = q_vec[k]
            # Bit 0 means +, Bit 1 means -
            if (mask & (1 << (k - 1))) == 0
                d = p + q
            else
                d = abs(p - q)
            end
            if d > 0
                cur_degs[v] = d
            end
        end
        m_idx = degrees_to_multiindex(cur_degs)
        terms[m_idx] = get(terms, m_idx, 0.0) + coeff
    end

    return [(k, v) for (k, v) in terms if abs(v) > 1e-15]
end

"""
    cheby_product(beta::Vector{UInt16}, gamma::Vector{UInt16}, item::Vector{UInt16}) -> Vector{Tuple{Vector{UInt16}, Float64}}

Computes three-way Chebyshev product T_beta * T_gamma * T_item.
"""
function cheby_product(beta::Vector{UInt16}, gamma::Vector{UInt16}, item::Vector{UInt16})::Vector{Tuple{Vector{UInt16}, Float64}}
    p1 = cheby_product(beta, gamma)
    if isempty(item)
        return p1
    end
    res = Dict{Vector{UInt16}, Float64}()
    for (m1, c1) in p1
        p2 = cheby_product(m1, item)
        for (m2, c2) in p2
            c_tot = c1 * c2
            res[m2] = get(res, m2, 0.0) + c_tot
        end
    end
    return [(k, v) for (k, v) in res if abs(v) > 1e-15]
end

"""
    univariate_monomial_to_chebyshev(deg::Int) -> Vector{Tuple{Int, Float64}}

Expands x^deg on [-1, 1] into a linear combination of Chebyshev polynomials T_k(x):
x^0 = T_0
x^1 = T_1
x^2 = 0.5 T_2 + 0.5 T_0
x^3 = 0.25 T_3 + 0.75 T_1
x^4 = 0.125 T_4 + 0.5 T_2 + 0.375 T_0
"""
function univariate_monomial_to_chebyshev(deg::Int)::Vector{Tuple{Int, Float64}}
    if deg == 0
        return [(0, 1.0)]
    elseif deg == 1
        return [(1, 1.0)]
    elseif deg == 2
        return [(2, 0.5), (0, 0.5)]
    elseif deg == 3
        return [(3, 0.25), (1, 0.75)]
    elseif deg == 4
        return [(4, 0.125), (2, 0.5), (0, 0.375)]
    else
        # Generic recurrence x * x^{d-1}
        prev = univariate_monomial_to_chebyshev(deg - 1)
        res = Dict{Int, Float64}()
        for (k, c) in prev
            # T_1 * T_k = 0.5 * (T_{k+1} + T_{|k-1|})
            if k == 0
                res[1] = get(res, 1, 0.0) + c
            else
                res[k + 1] = get(res, k + 1, 0.0) + 0.5 * c
                res[abs(k - 1)] = get(res, abs(k - 1), 0.0) + 0.5 * c
            end
        end
        return [(k, v) for (k, v) in res if abs(v) > 1e-15]
    end
end

"""
    monomial_to_chebyshev(mono::Vector{UInt16}, coeff::Float64=1.0) -> Vector{Tuple{Vector{UInt16}, Float64}}

Expands a multidimensional monomial x^alpha into a tensor Chebyshev polynomial series on [-1, 1]^n.
"""
function monomial_to_chebyshev(mono::Vector{UInt16}, coeff::Float64=1.0)::Vector{Tuple{Vector{UInt16}, Float64}}
    if isempty(mono)
        return [(UInt16[], coeff)]
    end
    degs = count_degrees(mono)
    # Start with scalar 1
    terms = Dict{Vector{UInt16}, Float64}(UInt16[] => coeff)

    for (v, d) in degs
        cheb_u = univariate_monomial_to_chebyshev(d)
        new_terms = Dict{Vector{UInt16}, Float64}()
        for (cur_mono, cur_c) in terms
            for (cheb_d, cheb_c) in cheb_u
                # Multiply cur_mono by T_{cheb_d}(x_v)
                v_mono = cheb_d > 0 ? UInt16[v for _ in 1:cheb_d] : UInt16[]
                prod_terms = cheby_product(cur_mono, v_mono)
                for (p_mono, p_c) in prod_terms
                    tot_c = cur_c * cheb_c * p_c
                    new_terms[p_mono] = get(new_terms, p_mono, 0.0) + tot_c
                end
            end
        end
        terms = new_terms
    end
    return [(k, v) for (k, v) in terms if abs(v) > 1e-15]
end

"""
    poly_to_chebyshev(p::poly{T}) -> poly{T}

Converts a polynomial `p` from standard monomial basis to the tensor Chebyshev basis on [-1, 1]^n.
"""
function poly_to_chebyshev(p::poly{T})::poly{Float64} where {T<:Number}
    cheb_dict = Dict{Vector{UInt16}, Float64}()
    for (i, mono) in enumerate(p.supp)
        c = Float64(p.coe[i])
        for (m, weight) in monomial_to_chebyshev(mono, c)
            cheb_dict[m] = get(cheb_dict, m, 0.0) + weight
        end
    end
    # Filter zeros and sort support
    supp = Vector{UInt16}[]
    coe = Float64[]
    for m in sort!(collect(keys(cheb_dict)))
        val = cheb_dict[m]
        if abs(val) > 1e-15
            push!(supp, m)
            push!(coe, val)
        end
    end
    if isempty(supp)
        supp = [UInt16[]]
        coe = [0.0]
    end
    return poly{Float64}(supp, coe)
end
