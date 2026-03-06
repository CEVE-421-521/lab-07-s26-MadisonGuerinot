#=
House Elevation Model for Robustness Analysis

Adapted from Lab 6 with additions:
- EAD emulator for fast evaluation across many SLR trajectories
- Dropped SimOptDecisions dependency (not needed for this lab)
- Same house parameters, depth-damage, and elevation cost functions

References:
- Zarekarizi et al. (2020): House elevation cost model
- Ruckert et al. (2019): BRICK sea-level projections
=#

using DataFrames
using Distributions
using Interpolations

# ============================================================
# Depth-Damage Function
# ============================================================

struct DepthDamageFunction
    itp::Interpolations.AbstractExtrapolation
end

function DepthDamageFunction(depths_ft::Vector{Float64}, damages_pct::Vector{Float64})
    order = sortperm(depths_ft)
    return DepthDamageFunction(
        LinearInterpolation(depths_ft[order], damages_pct[order]; extrapolation_bc=Flat())
    )
end

"""Return damage as a percentage of house value for a given flood depth (ft)."""
(ddf::DepthDamageFunction)(depth_ft::Real) = ddf.itp(Float64(depth_ft))

# ============================================================
# House
# ============================================================

struct House
    value_usd::Float64
    area_ft2::Float64
    height_above_gauge_ft::Float64
    ddf::DepthDamageFunction
end

"""Construct a House from a HAZUS depth-damage row and physical parameters."""
function House(row::DataFrameRow; value_usd, area_ft2, height_above_gauge_ft)
    depths_ft = Float64[]
    damages_pct = Float64[]
    for (col, val) in pairs(row)
        col_str = string(col)
        startswith(col_str, "ft") || continue
        string(val) == "NA" && continue
        depth_str = col_str[3:end]
        is_neg = endswith(depth_str, "m")
        depth_str = is_neg ? depth_str[1:end-1] : depth_str
        depth_str = replace(depth_str, "_" => ".")
        d = parse(Float64, depth_str)
        push!(depths_ft, is_neg ? -d : d)
        push!(damages_pct, parse(Float64, string(val)))
    end
    ddf = DepthDamageFunction(depths_ft, damages_pct)
    return House(Float64(value_usd), Float64(area_ft2), Float64(height_above_gauge_ft), ddf)
end

# ============================================================
# Elevation Cost (Zarekarizi et al. 2020)
# ============================================================

const ELEVATION_COST_ITP = let
    thresholds = [0.0, 5.0, 8.5, 12.0, 14.0]
    rates_per_sqft = [80.36, 82.5, 86.25, 103.75, 113.75]
    LinearInterpolation(thresholds, rates_per_sqft)
end

"""Cost (USD) to elevate a house by Δh_ft feet. Returns 0 for no elevation."""
function elevation_cost(house::House, Δh_ft::Real)
    Δh_ft ≈ 0.0 && return 0.0
    (Δh_ft < 0 || Δh_ft > 14) && error("Elevation must be between 0 and 14 ft, got $Δh_ft")
    base_cost = 10_000 + 300 + 470 + 4_300 + 2_175 + 3_500  # $20,745
    return base_cost + house.area_ft2 * ELEVATION_COST_ITP(Δh_ft)
end

# ============================================================
# Expected Annual Damage (trapezoidal integration)
# ============================================================

"""
Compute EAD via trapezoidal integration over the exceedance-probability curve.
`clearance_ft` is the height of the house floor above current sea level (including SLR and elevation).
"""
function expected_annual_damage(house::House, clearance_ft::Real, surge_dist)
    n = 1000
    p_exceed = range(0.0001, 0.9999; length=n)
    flood_depths = quantile.(Ref(surge_dist), 1.0 .- collect(p_exceed))
    net_depths = flood_depths .- clearance_ft
    damages_usd = (house.ddf.(net_depths) ./ 100.0) .* house.value_usd
    ead = sum(
        (damages_usd[i] + damages_usd[i + 1]) / 2 * (p_exceed[i + 1] - p_exceed[i])
        for i in 1:(n - 1)
    )
    return ead
end

# ============================================================
# EAD Emulator
# ============================================================

"""
Build an EAD emulator: a fast interpolation that maps clearance (ft) to EAD (USD).
Pre-computes EAD at a grid of clearance values and interpolates between them.
This avoids recomputing the trapezoidal integration for every (trajectory, year) pair.
"""
function build_ead_emulator(house::House, surge_dist; clearance_range=(-10.0, 30.0), n_points=500)
    clearances = range(clearance_range[1], clearance_range[2]; length=n_points)
    eads = [expected_annual_damage(house, c, surge_dist) for c in clearances]
    itp = LinearInterpolation(collect(clearances), eads; extrapolation_bc=Flat())
    return itp
end
