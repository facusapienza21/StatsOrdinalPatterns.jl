export test_op_majorization, OPMajorizationTestResult

# ------------------------------------------------------------------------------
# 1. HELPERS
# ------------------------------------------------------------------------------

# Partial sums of the probabilities sorted in decreasing order. `p` majorizes `q` when the
# curve of `p` lies above the curve of `q` at every index.
# This may be replaced by `ComplexityMeasures.lorenz_curve` once it is released.
_lorenz_curve(p::AbstractVector{<:Real}) = cumsum(sort(p; rev=true))

# Lehmer codes of the ordinal patterns of `data` with pattern length `m` and delay `d`,
# one code per pattern, in the order of the series.
function _op_codes(data::AbstractVector{<:Real}, m::Int, d::Int)
  n_patterns = length(data) - (m - 1) * d
  n_patterns >= 1 || throw(ArgumentError(
    "the series has $(length(data)) observations, too few for patterns of length " *
    "$m with delay $d."
  ))
  codes = Vector{Int}(undef, n_patterns)
  win = zeros(Int, m)
  idx_used = zeros(Int, m)
  for i in 1:n_patterns
    sortperm!(win, view(data, i:d:(i+(m-1)*d)))
    fill!(idx_used, 0)
    codes[i] = perm_to_lehm_idx!(win, idx_used)
  end
  return codes
end

# Relative frequencies of the codes `1:length(p)` in `codes`, written into `p`.
function _code_frequencies!(p::Vector{Float64}, codes::Vector{Int})
  fill!(p, 0.0)
  @inbounds for c in codes
    p[c] += 1.0
  end
  p ./= length(codes)
  return p
end

# Moving-block resampling of a code sequence into `out`, same scheme as `bootstrap_op`.
function _block_resample!(out::Vector{Int}, codes::Vector{Int}, block_size::Int)
  n = length(codes)
  i = 1
  @inbounds while i <= n
    start = rand(1:(n-block_size+1))
    for j in 0:(block_size-1)
      i > n && break
      out[i] = codes[start+j]
      i += 1
    end
  end
  return out
end

# Relative frequencies of one multinomial resample of `n` patterns with probabilities
# `p`, written into `out`. Equivalent to an i.i.d. resample of the pattern sequence.
function _multinomial_resample!(out::Vector{Float64}, p::Vector{Float64}, n::Int)
  rand!(Multinomial(n, p), out)
  out ./= n
  return out
end

# ------------------------------------------------------------------------------
# 2. BOOTSTRAP OF THE GAPS BETWEEN THE TWO LORENZ CURVES
# ------------------------------------------------------------------------------
# Both methods return an `n_boot × (n_pat - 1)` matrix whose row `b` holds the resampled
# gaps S_k(p*) - S_k(q*) for k = 1, …, n_pat - 1. The last partial sum of each curve is
# always 1, so the last gap is dropped.

# Gaps of the Lorenz curves of `p` and `q`, written into row `b` of `gaps`.
function _lorenz_gaps!(gaps::Matrix{Float64}, b::Int, p::Vector{Float64},
  q::Vector{Float64})
  S_p = _lorenz_curve(p)
  S_q = _lorenz_curve(q)
  @inbounds for k in 1:size(gaps, 2)
    gaps[b, k] = S_p[k] - S_q[k]
  end
  return gaps
end

# i.i.d. patterns (`block_size == 1`): multinomial resampling of the pattern counts from
# the observed frequencies `p` and `q`, with `n_p` and `n_q` patterns respectively.
function _bootstrap_gaps(p::Vector{Float64}, q::Vector{Float64}, n_p::Int, n_q::Int,
  n_boot::Int)
  n_pat = length(p)
  gaps = Matrix{Float64}(undef, n_boot, n_pat - 1)
  p_star = similar(p)
  q_star = similar(q)
  for b in 1:n_boot
    _multinomial_resample!(p_star, p, n_p)
    _multinomial_resample!(q_star, q, n_q)
    _lorenz_gaps!(gaps, b, p_star, q_star)
  end
  return gaps
end

# Serially dependent patterns (`block_size > 1`): moving-block resampling of the two code
# sequences, then counting. Cheaper than resampling the raw series and re-encoding, and
# it does not create spurious patterns at the block joints.
function _bootstrap_gaps(codes_x::Vector{Int}, codes_y::Vector{Int}, n_pat::Int,
  n_boot::Int, block_size::Int)
  gaps = Matrix{Float64}(undef, n_boot, n_pat - 1)
  x_star = similar(codes_x)
  y_star = similar(codes_y)
  p_star = zeros(n_pat)
  q_star = zeros(n_pat)
  for b in 1:n_boot
    _block_resample!(x_star, codes_x, block_size)
    _block_resample!(y_star, codes_y, block_size)
    _code_frequencies!(p_star, x_star)
    _code_frequencies!(q_star, y_star)
    _lorenz_gaps!(gaps, b, p_star, q_star)
  end
  return gaps
end

# ------------------------------------------------------------------------------
# 3. RESULT TYPE
# ------------------------------------------------------------------------------
"""
    OPMajorizationTestResult

Result of the bootstrap majorization test [`test_op_majorization`](@ref).

Fields:
- `stat::Float64`: value of the test statistic, the smallest studentised gap between the
  two Lorenz curves over the non-degenerate constraints. Negative values are evidence
  against H₀.
- `boot_crit::Float64`: bootstrap critical value (lower tail).
- `boot_pval::Float64`: bootstrap p-value.
- `boot_reject::Bool`: whether H₀ (the first distribution majorizes the second) is
  rejected at the chosen level.
- `n_boot::Int`: number of bootstrap replications.
- `boot_dist::Vector{Float64}`: the `n_boot` resampled statistics, i.e. the bootstrap
  null distribution the critical value and the p-value were computed from.
- `gaps::Vector{Float64}`: observed gaps ``S_k(\\hat p) - S_k(\\hat q)``,
  ``k = 1, …, m! - 1``, between the two Lorenz curves.
- `sigma::Vector{Float64}`: bootstrap standard error of each gap.
- `selected::Vector{Int}`: indices of the near-binding constraints that enter the
  bootstrap minimum.
- `n_eff::Float64`: effective sample size used for the selection threshold.
- `block_size::Int`: block length of the resampling, `1` for multinomial resampling.

With Makie loaded, `plot(res)` draws a histogram of `boot_dist` with the rejection region,
the critical value and the observed statistic; see the plotting tutorial in the docs.
"""
struct OPMajorizationTestResult
  stat::Float64
  boot_crit::Float64
  boot_pval::Float64
  boot_reject::Bool
  n_boot::Int
  boot_dist::Vector{Float64}
  gaps::Vector{Float64}
  sigma::Vector{Float64}
  selected::Vector{Int}
  n_eff::Float64
  block_size::Int
end

function Base.show(io::IO, r::OPMajorizationTestResult)
  println(io, "OPMajorizationTestResult")
  println(io, "  H₀:               p ≻ q (first distribution majorizes the second)")
  println(io, "  Statistic:        ", round(r.stat, digits=4))
  println(io, "  Constraints:      ", length(r.selected), " of ", length(r.gaps), " selected")
  println(io, "  ─────────────────────────────")
  println(io, "  Bootstrap  (n_boot = ", r.n_boot, ", block_size = ", r.block_size, ")")
  println(io, "    Critical value: ", round(r.boot_crit, digits=4))
  println(io, "    p-value:        ", round(r.boot_pval, digits=4))
  print(io, "    Reject H₀:      ", r.boot_reject)
end

# ------------------------------------------------------------------------------
# 4. INFERENCE FROM THE OBSERVED CURVES AND THE BOOTSTRAPPED GAPS
# ------------------------------------------------------------------------------
# Shared by both resampling schemes. `S_p`, `S_q` are the observed Lorenz curves and
# `gaps_boot` the matrix from `_bootstrap_gaps`. The test is the generalized moment
# selection (GMS) test of Andrews and Soares (2010) for the moment inequalities
# S_k(p) - S_k(q) >= 0, with a bootstrap in place of the asymptotic distribution.
function _majorization_inference(S_p::Vector{Float64}, S_q::Vector{Float64},
  gaps_boot::Matrix{Float64}, n_eff::Float64, alpha::Float64, c::Float64, block_size::Int)
  n_boot, K = size(gaps_boot)
  gaps = S_p[1:K] .- S_q[1:K]

  # Bootstrap standard error of each gap, floored at the scale 1/n_eff so that
  # constraints with nearly degenerate partial sums do not inflate the ratios.
  sigma = max.(vec(std(gaps_boot; dims=1)), 1.0 / n_eff)

  # Constraints at which both curves have already reached 1 (up to 1/n_eff) are a
  # difference of two near-cancelling numbers and carry no information: excluded.
  thresh = 1.0 - 1.0 / n_eff
  valid = findall(k -> !(S_p[k] >= thresh && S_q[k] >= thresh), 1:K)

  if isempty(valid)
    # Both distributions are degenerate: nothing to test, H₀ is not rejected.
    return OPMajorizationTestResult(0.0, 0.0, 1.0, false, n_boot, zeros(n_boot), gaps,
      sigma, Int[], n_eff, block_size)
  end

  z = gaps[valid] ./ sigma[valid]
  stat = minimum(z)

  # GMS: only the near-binding constraints enter the bootstrap minimum. The threshold
  # must grow with the sample size, but slower than its square root.
  kappa = c * sqrt(log(n_eff))
  sel = findall(<=(kappa), z)
  isempty(sel) && (sel = collect(eachindex(z)))
  selected = valid[sel]

  # Bootstrap distribution of the minimum of the recentred, studentised gaps
  boot_dist = Vector{Float64}(undef, n_boot)
  @inbounds for b in 1:n_boot
    t = Inf
    for k in selected
      t = min(t, (gaps_boot[b, k] - gaps[k]) / sigma[k])
    end
    boot_dist[b] = t
  end

  boot_crit = quantile(boot_dist, alpha)
  boot_pval = count(<=(stat), boot_dist) / n_boot
  boot_reject = stat < boot_crit
  return OPMajorizationTestResult(stat, boot_crit, boot_pval, boot_reject, n_boot,
    boot_dist, gaps, sigma, selected, n_eff, block_size)
end

# ------------------------------------------------------------------------------
# 5. MAJORIZATION TEST
# ------------------------------------------------------------------------------
"""
    test_op_majorization(x, y, n_boot; method=:multinomial, m=3, d=1, block_size=nothing,
                         alpha=0.05, c=1.0)
    test_op_majorization(p, q, n_p, n_q, n_boot; alpha=0.05, c=1.0)

Bootstrap test of H₀: the ordinal pattern distribution of `x` majorizes that of `y`,
against H₁: it does not. Return an [`OPMajorizationTestResult`](@ref).

In the first form `x` and `y` are time series, encoded into ordinal patterns of length
`m` with delay `d`; they need not have the same length. In the second form `p` and `q`
are the two estimated pattern distributions, vectors of the same length summing to one,
and `n_p`, `n_q` the numbers of patterns they were estimated from. The second form always
uses multinomial resampling, since the pattern sequences are not available.

A distribution `p` majorizes `q`, written ``p \\succ q``, when the partial sums of `p`
sorted in decreasing order dominate those of `q` at every index,
``S_k(p) \\geq S_k(q)`` for ``k = 1, …, m!``: `p` is at least as concentrated as `q`.
Additive observational noise acts on the pattern distribution through a doubly
stochastic map, so the distribution of a noiseless series majorizes that of its noisy
version. The test is one sided: rejection says that `x` is not the cleaner of the two.

- `n_boot`: number of bootstrap replications.
- `method`: resampling scheme, first form only.
  - `:multinomial` (default): the patterns are treated as i.i.d. and the pattern counts
    of each series are resampled from a multinomial distribution.
  - `:blockbootstrap`: the two pattern sequences are resampled with a moving-block
    bootstrap of length `block_size`, which preserves the serial dependence between
    consecutive patterns. `block_size` is required with this method and must not be
    given with `:multinomial`.
- `alpha`: significance level (default `0.05`).
- `c`: scale of the constraint selection threshold, see below.

# Details
The statistic is ``T = \\min_k (S_k(\\hat p) - S_k(\\hat q)) / \\hat\\sigma_k`` over the
non-degenerate constraints, with ``\\hat\\sigma_k`` the bootstrap standard error of the
``k``-th gap. Negative values are evidence against H₀. Following the generalized moment
selection principle of Andrews and Soares (2010), only the near-binding constraints with
``(S_k(\\hat p) - S_k(\\hat q)) / \\hat\\sigma_k \\leq c \\sqrt{\\log n_{\\mathrm{eff}}}``
enter the bootstrap minimum, where ``n_{\\mathrm{eff}} = n_p n_q / (n_p + n_q)`` is the
harmonic mean of the two numbers of patterns. Constraints at which both curves have
reached one up to ``1 / n_{\\mathrm{eff}}`` are excluded. The bootstrap distribution is
that of the recentred minimum over the selected constraints, and the test rejects for
small values of ``T``. At the boundary of H₀ (equal distributions) the test attains its
nominal level; away from it, it is conservative.

With block resampling, ``n_{\\mathrm{eff}}`` is still computed from the numbers of
patterns, so the selection threshold does not account for the reduced information in
dependent patterns.

# References
- Andrews, D. W. K. and Soares, G. (2010). Inference for parameters defined by moment
  inequalities using generalized moment selection. *Econometrica*, 78(1), 119–157.
- Linton, O., Maasoumi, E. and Whang, Y.-J. (2005). Consistent testing for stochastic
  dominance under general sampling schemes. *Review of Economic Studies*, 72(3), 735–765.

# Examples
```julia
x = sin.(range(0, 20π; length=2000))
y = x .+ 0.5 .* randn(2000)          # noisy version: the patterns of x majorize those of y
test_op_majorization(x, y, 999; m=3)  # H₀ holds, not rejected
test_op_majorization(y, x, 999; m=3)  # reverse hypothesis, rejected
test_op_majorization(x, y, 999; m=3, method=:blockbootstrap, block_size=20)

# From estimated pattern distributions and their numbers of patterns
test_op_majorization([0.5, 0.3, 0.2], [0.4, 0.35, 0.25], 500, 500, 999)
```
"""
function test_op_majorization(
  x::AbstractVector{<:Real},
  y::AbstractVector{<:Real},
  n_boot::Int;
  method::Symbol=:multinomial,
  m::Int=3,
  d::Int=1,
  block_size::Union{Nothing,Int}=nothing,
  alpha::Float64=0.05,
  c::Float64=1.0
)
  n_boot >= 1 || throw(ArgumentError("n_boot must be at least 1, got $n_boot."))
  if method == :multinomial
    isnothing(block_size) || throw(ArgumentError(
      "block_size applies to method=:blockbootstrap only."
    ))
  elseif method == :blockbootstrap
    isnothing(block_size) && throw(ArgumentError(
      "block_size is required with method=:blockbootstrap."
    ))
    block_size >= 1 || throw(ArgumentError("block_size must be at least 1, got $block_size."))
  else
    throw(ArgumentError("method must be :multinomial or :blockbootstrap, got :$method."))
  end

  n_pat = factorial(m)
  codes_x = _op_codes(x, m, d)
  codes_y = _op_codes(y, m, d)
  p = _code_frequencies!(zeros(n_pat), codes_x)
  q = _code_frequencies!(zeros(n_pat), codes_y)
  n_x, n_y = length(codes_x), length(codes_y)
  n_eff = n_x * n_y / (n_x + n_y)

  if method == :multinomial
    gaps_boot = _bootstrap_gaps(p, q, n_x, n_y, n_boot)
    block_size = 1
  else
    gaps_boot = _bootstrap_gaps(codes_x, codes_y, n_pat, n_boot, block_size)
  end
  return _majorization_inference(_lorenz_curve(p), _lorenz_curve(q), gaps_boot, n_eff,
    alpha, c, block_size)
end

function test_op_majorization(
  p::AbstractVector{<:Real},
  q::AbstractVector{<:Real},
  n_p::Int,
  n_q::Int,
  n_boot::Int;
  alpha::Float64=0.05,
  c::Float64=1.0
)
  n_boot >= 1 || throw(ArgumentError("n_boot must be at least 1, got $n_boot."))
  length(p) == length(q) || throw(ArgumentError(
    "p and q must have the same length, got $(length(p)) and $(length(q))."
  ))
  length(p) >= 2 || throw(ArgumentError("p and q must have at least two entries."))
  (sum(p) ≈ 1 && sum(q) ≈ 1) || throw(ArgumentError("p and q must each sum to one."))
  (n_p >= 1 && n_q >= 1) || throw(ArgumentError("n_p and n_q must be at least 1."))

  pv = Vector{Float64}(p)
  qv = Vector{Float64}(q)
  n_eff = n_p * n_q / (n_p + n_q)
  gaps_boot = _bootstrap_gaps(pv, qv, n_p, n_q, n_boot)
  return _majorization_inference(_lorenz_curve(pv), _lorenz_curve(qv), gaps_boot, n_eff,
    alpha, c, 1)
end
