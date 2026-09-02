"""Alternative source-coverage models for Spectro's Python backend.

The functions in this module transform intrinsic optical-depth arrays into
normalized flux. Instrumental convolution must be applied *after* this
transformation.

Power-law convention
--------------------
By default the power-law mode interprets Spectro's Voigt optical depth as the
area-averaged optical depth. For ``tau(x) = tau_max*x**a`` this gives
``tau_max = (a + 1) * tau_mean``. This is a convenient parameterization, but
it does *not* make the raw fitted ``N`` directly comparable to the legacy PPC
``N``: in PPC the Voigt optical depth is the optical depth inside the covered
portion. Model-to-model column comparisons should therefore use the appropriate
spatially averaged optical depth, not the raw parameter value.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping
import time

import numpy as np
from scipy.special import gammainc, gammaln, hyp1f1


COVERAGE_MODE_NAMES = (
    "full",
    "ppc",
    "powerlaw",          # historical x/area law; mode id 2
    "powerlaw_ppc",      # historical x/area law + uncovered; mode id 3
    "two_zone",
    "cfov",
    "powerlaw_radial",
    "powerlaw_radial_ppc",
    "powerlaw_ppc_sysa",  # x-law: per-system a + per-LFR uncovered fraction; mode id 8
    "powerlaw_sysa_global",  # global x-law: one a per system, all transitions; Cf=1; mode id 9
)

COVERAGE_MODE_LABELS = (
    "Full coverage",
    "Partial covering",
    "Power-law inhomogeneous (x)",
    "Power-law + uncovered (x)",
    "Two-zone",
    "Overlap Cf (2 components)",
    "Radial power-law inhomogeneous",
    "Radial power-law + uncovered",
    "Power-law + uncovered (x; system a)",
    "Global system x-PL (all transitions; Cf=1)",
)


def coverage_mode_name(value) -> str:
    """Return a valid model name from a stored numeric/string value."""
    if isinstance(value, str):
        value = value.strip().lower()
        if value in COVERAGE_MODE_NAMES:
            return value
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            return "ppc"
    try:
        index = int(round(float(value)))
    except (TypeError, ValueError):
        return "ppc"
    return COVERAGE_MODE_NAMES[index] if 0 <= index < len(COVERAGE_MODE_NAMES) else "ppc"


def coverage_mode_index(value) -> int:
    return COVERAGE_MODE_NAMES.index(coverage_mode_name(value))


def _tau(tau):
    arr = np.asarray(tau, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Optical depth contains non-finite values")
    if np.any(arr < 0):
        raise ValueError("Optical depth must be non-negative")
    return arr


def _fraction(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return value


def full_coverage(tau):
    return np.exp(-_tau(tau))


def partial_covering(tau, cf):
    """Homogeneous partial covering: I = (1-Cf) + Cf*exp(-tau)."""
    tau = _tau(tau)
    cf = _fraction(cf, "cf")
    return (1.0 - cf) + cf * np.exp(-tau)


def overlap_bounds(cf1, cf2):
    cf1 = _fraction(cf1, "cf1")
    cf2 = _fraction(cf2, "cf2")
    return max(0.0, cf1 + cf2 - 1.0), min(cf1, cf2)


def overlap_from_normalized(cf1, cf2, cfov_norm):
    q = _fraction(cfov_norm, "cfov_norm")
    lower, upper = overlap_bounds(cf1, cf2)
    return lower + q * (upper - lower)


def overlap_partial_covering(tau1, tau2, cf1, cf2, cfov_norm):
    """Two partially covering absorbers with arbitrary projected overlap."""
    tau1, tau2 = np.broadcast_arrays(_tau(tau1), _tau(tau2))
    cf1 = _fraction(cf1, "cf1")
    cf2 = _fraction(cf2, "cf2")
    cfov = overlap_from_normalized(cf1, cf2, cfov_norm)

    uncovered = 1.0 - cf1 - cf2 + cfov
    only1 = cf1 - cfov
    only2 = cf2 - cfov
    flux = (
        uncovered
        + only1 * np.exp(-tau1)
        + only2 * np.exp(-tau2)
        + cfov * np.exp(-(tau1 + tau2))
    )
    return np.clip(flux, 0.0, 1.0)


@lru_cache(maxsize=8)
def _legendre_nodes(n_nodes):
    if int(n_nodes) < 16:
        raise ValueError("At least 16 quadrature nodes are required")
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    return (x + 1.0) / 2.0, w / 2.0


def _powerlaw_moment(tau_max, a, moment=0, *, n_nodes=48, profile=None, profile_label=None):
    r"""Return ``J_m = integral_0^1 u^m exp(-tau_max*u^a) du``.

    The regularized incomplete-gamma representation is the primary evaluator:

        J_m = gamma(s, tau_max) / [a * tau_max**s],
        s = (m + 1) / a.

    This avoids the severe ``hyp1f1`` slowdowns seen during real Spectro fits
    around ``a ~ 10--20`` and at large optical depth.  ``hyp1f1`` is retained
    only as a numerical rescue path for points where the regularized-gamma
    evaluation under/overflows, and cached Gauss-Legendre quadrature remains a
    final fallback.
    """
    total_started = time.perf_counter()
    tau_max = _tau(tau_max)
    a = float(a)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("power-law exponent a must be positive")
    m = float(moment)
    if m < 0:
        raise ValueError("moment must be non-negative")

    shape = (m + 1.0) / a
    zero_value = 1.0 / (m + 1.0)
    value = np.full_like(tau_max, zero_value, dtype=float)

    positive = tau_max > 1e-12
    gamma_failed = np.zeros_like(tau_max, dtype=bool)
    gamma_seconds = 0.0
    hyp_seconds = 0.0
    quadrature_seconds = 0.0
    n_hyp = 0
    n_quad = 0
    if np.any(positive):
        t = tau_max[positive]
        stage_started = time.perf_counter()
        reg_lower = gammainc(shape, t)
        gamma_ok = np.isfinite(reg_lower) & (reg_lower > 0.0)
        local_value = np.full_like(t, np.nan, dtype=float)
        if np.any(gamma_ok):
            tg = t[gamma_ok]
            pg = reg_lower[gamma_ok]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                log_value = (
                    gammaln(shape) + np.log(pg) - np.log(a)
                    - shape * np.log(tg)
                )
                local_value[gamma_ok] = np.exp(log_value)
        gamma_seconds += time.perf_counter() - stage_started
        value[positive] = local_value
        failed_local = ~gamma_ok
        if np.any(failed_local):
            positive_indices = np.flatnonzero(positive)
            gamma_failed.flat[positive_indices[failed_local]] = True

    bad = (gamma_failed | (~np.isfinite(value)) | (value < 0.0)
           | (value > zero_value * (1.0 + 1e-10)))

    if np.any(bad):
        n_hyp = int(np.count_nonzero(bad))
        stage_started = time.perf_counter()
        rescued = np.asarray(
            hyp1f1(shape, shape + 1.0, -tau_max[bad]) / (m + 1.0),
            dtype=float,
        )
        hyp_seconds += time.perf_counter() - stage_started
        value[bad] = rescued

    bad = ((~np.isfinite(value)) | (value < 0.0)
           | (value > zero_value * (1.0 + 1e-10)))
    if np.any(bad):
        n_quad = int(np.count_nonzero(bad))
        stage_started = time.perf_counter()
        u, w = _legendre_nodes(max(int(n_nodes), 64))
        tau_bad = np.asarray(tau_max[bad], dtype=float).reshape(-1)
        fallback = np.sum(
            (u ** m) * np.exp(-tau_bad[:, None] * u**a) * w,
            axis=-1,
        )
        quadrature_seconds += time.perf_counter() - stage_started
        value[bad] = fallback

    result = np.clip(value, 0.0, zero_value)
    if profile is not None:
        finite_tau = tau_max[np.isfinite(tau_max)]
        profile.setdefault("powerlaw_moments", []).append({
            "label": profile_label or f"J{int(m) if m.is_integer() else m}",
            "moment": m,
            "a": a,
            "shape": shape,
            "n_points": int(tau_max.size),
            "n_positive": int(np.count_nonzero(positive)),
            "n_hyp_rescue": n_hyp,
            "n_quad_fallback": n_quad,
            "tau_min": float(np.min(finite_tau)) if finite_tau.size else float("nan"),
            "tau_max": float(np.max(finite_tau)) if finite_tau.size else float("nan"),
            "gamma_seconds": gamma_seconds,
            "hyp_seconds": hyp_seconds,
            "quadrature_seconds": quadrature_seconds,
            "total_seconds": time.perf_counter() - total_started,
        })
    return result


def powerlaw_x_inhomogeneous(tau, log10_a, *, tau_kind="mean", n_nodes=48, profile=None):
    r"""Arav/de Kool x/strip-coordinate power-law distribution.

        tau(x) = tau_max x^a, 0 <= x <= 1,
        I = integral_0^1 exp[-tau(x)] dx.

    For ``tau_kind='mean'``, ``tau_max=(a+1)<tau>``.
    """
    tau = _tau(tau)
    log10_a = float(log10_a)
    if not np.isfinite(log10_a):
        raise ValueError("log10_a must be finite")
    a = 10.0 ** log10_a
    if tau_kind == "mean":
        tau_max = (a + 1.0) * tau
    elif tau_kind == "max":
        tau_max = tau
    else:
        raise ValueError("tau_kind must be 'mean' or 'max'")
    return np.clip(_powerlaw_moment(tau_max, a, 0, n_nodes=n_nodes, profile=profile, profile_label="J0"), 0.0, 1.0)


def powerlaw_radial_inhomogeneous(tau, log10_a, *, tau_kind="mean", n_nodes=48, profile=None):
    r"""Genuinely radial cloud with optical depth decreasing toward its edge.

    With ``rho=r/R_abs`` we use

        tau(rho) = tau_max (1-rho)^a, 0 <= rho <= 1.

    The circular area average is

        I = 2 integral_0^1 rho exp[-tau_max(1-rho)^a] d rho
          = 2 [J_0 - J_1].

    The area-mean optical depth is

        <tau> = 2 tau_max / [(a+1)(a+2)],

    hence ``tau_max=<tau>(a+1)(a+2)/2`` for ``tau_kind='mean'``.
    """
    tau = _tau(tau)
    log10_a = float(log10_a)
    if not np.isfinite(log10_a):
        raise ValueError("log10_a must be finite")
    a = 10.0 ** log10_a
    if tau_kind == "mean":
        tau_max = 0.5 * (a + 1.0) * (a + 2.0) * tau
    elif tau_kind == "max":
        tau_max = tau
    else:
        raise ValueError("tau_kind must be 'mean' or 'max'")

    j0 = _powerlaw_moment(tau_max, a, 0, n_nodes=n_nodes, profile=profile, profile_label="J0")
    j1 = _powerlaw_moment(tau_max, a, 1, n_nodes=n_nodes, profile=profile, profile_label="J1")
    transmission = 2.0 * (j0 - j1)
    bad = (~np.isfinite(transmission)) | (transmission < 0.0) | (transmission > 1.0)
    if np.any(bad):
        rho, w = _legendre_nodes(max(int(n_nodes), 64))
        fallback = np.sum(
            2.0 * rho * np.exp(-tau_max[..., None] * (1.0-rho)**a) * w,
            axis=-1,
        )
        transmission = np.where(bad, fallback, transmission)
    return np.clip(transmission, 0.0, 1.0)


def powerlaw_inhomogeneous(tau, log10_a, *, tau_kind="mean", n_nodes=48, profile=None):
    return powerlaw_x_inhomogeneous(
        tau, log10_a, tau_kind=tau_kind, n_nodes=n_nodes, profile=profile
    )


def powerlaw_with_uncovered_fraction(
    tau, cf, log10_a, *, tau_kind="mean", n_nodes=48, geometry="radial", profile=None
):
    cf = _fraction(cf, "cf")
    if geometry == "radial":
        covered = powerlaw_radial_inhomogeneous(
            tau, log10_a, tau_kind=tau_kind, n_nodes=n_nodes, profile=profile
        )
    elif geometry == "x":
        covered = powerlaw_x_inhomogeneous(
            tau, log10_a, tau_kind=tau_kind, n_nodes=n_nodes, profile=profile
        )
    else:
        raise ValueError("geometry must be 'radial' or 'x'")
    return (1.0 - cf) + cf * covered

def powerlaw_system_product_with_uncovered(
    tau_by_system: Mapping[int, np.ndarray], cf, log10_a_by_system, *, profile=None
):
    r"""Component-specific x-power laws wrapped by one LFR uncovered fraction.

    Each kinematic system keeps its own historical x-law exponent ``a_k``.
    Inside the geometrically covered fraction the system transmissions multiply,

        T_cov = product_k P_x(tau_k, a_k),
        I = (1-Cf) + Cf*T_cov.

    Therefore ``Cf=1`` is exactly nested in the historical component-specific
    x-PL construction obtained by multiplying one x-PL transmission per system.
    The ``Cf`` value remains an LFR/transition parameter, while the exponents are
    attached to velocity-system ids and are shared by every wavelength region.
    """
    cf = _fraction(cf, "cf")
    system_ids = sorted(tau_by_system)
    if not system_ids:
        raise ValueError("tau_by_system is empty")

    first = _tau(tau_by_system[system_ids[0]])
    covered = np.ones_like(first, dtype=float)
    for system_id in system_ids:
        if system_id not in log10_a_by_system:
            raise ValueError(
                f"Missing system power-law exponent for sys{system_id}; "
                f"expected covasys_{system_id}"
            )
        covered *= powerlaw_x_inhomogeneous(
            tau_by_system[system_id],
            log10_a_by_system[system_id],
            profile=profile,
        )
    return (1.0 - cf) + cf * covered


def two_zone(tau_high, cf_high, tau_ratio):
    """Two optical-depth zones; PPC is recovered when ``tau_ratio=0``."""
    tau_high = _tau(tau_high)
    cf_high = _fraction(cf_high, "cf_high")
    tau_ratio = _fraction(tau_ratio, "tau_ratio")
    return (
        (1.0 - cf_high) * np.exp(-tau_ratio * tau_high)
        + cf_high * np.exp(-tau_high)
    )


def region_transmission(mode, tau_by_system: Mapping[int, np.ndarray], params, profile=None):
    """Apply one coverage model to the optical depth in one Cf region.

    Parameters are plain floats. For ``cfov`` exactly two system arrays are
    supported in the total model. A one-system call is accepted for drawing
    individual component curves and reduces to ordinary partial covering.
    """
    mode = coverage_mode_name(mode)
    if not tau_by_system:
        raise ValueError("tau_by_system is empty")

    system_ids = sorted(tau_by_system)
    tau_total = np.sum([_tau(tau_by_system[i]) for i in system_ids], axis=0)

    if mode == "full":
        return full_coverage(tau_total)
    if mode == "ppc":
        return partial_covering(tau_total, params["cf"])
    if mode == "powerlaw":
        return powerlaw_x_inhomogeneous(tau_total, params["log10_a"], profile=profile)
    if mode == "powerlaw_ppc":
        return powerlaw_with_uncovered_fraction(
            tau_total, params["cf"], params["log10_a"], geometry="x", profile=profile
        )
    if mode == "powerlaw_ppc_sysa":
        return powerlaw_system_product_with_uncovered(
            tau_by_system, params["cf"], params["log10_a_by_system"], profile=profile
        )
    if mode == "powerlaw_sysa_global":
        return powerlaw_system_product_with_uncovered(
            tau_by_system, 1.0, params["log10_a_by_system"], profile=profile
        )
    if mode == "powerlaw_radial":
        return powerlaw_radial_inhomogeneous(tau_total, params["log10_a"], profile=profile)
    if mode == "powerlaw_radial_ppc":
        return powerlaw_with_uncovered_fraction(
            tau_total, params["cf"], params["log10_a"], geometry="radial", profile=profile
        )
    if mode == "two_zone":
        return two_zone(tau_total, params["cf"], params["tau_ratio"])
    if mode == "cfov":
        if len(system_ids) == 1:
            cf = params["cf1"] if system_ids[0] == 0 else params["cf2"]
            return partial_covering(tau_total, cf)
        if len(system_ids) != 2:
            raise ValueError(
                "The overlap-Cf model currently supports exactly two active "
                f"systems per covering region; found {len(system_ids)}"
            )
        return overlap_partial_covering(
            tau_by_system[system_ids[0]],
            tau_by_system[system_ids[1]],
            params["cf1"],
            params["cf2"],
            params["cfov_norm"],
        )
    raise ValueError(f"Unknown coverage model: {mode}")


def parse_coverage_assignment(addinfo):
    """Parse legacy ``cf.addinfo`` into system ids and exposure selector.

    Examples accepted by historical Spectro files include ``all_all``,
    ``sys0_all``, ``sys0sys1_exp0`` and ``sys0_exp_0``.  The returned exposure
    is ``None`` for all exposures or an integer otherwise.
    """
    text = str(addinfo or "all").strip()
    parts = text.split("_")
    system_part = parts[0] if parts and parts[0] else "all"
    if system_part == "all":
        systems = None
    else:
        import re
        systems = tuple(int(v) for v in re.findall(r"sys(\d+)", system_part))
        if not systems:
            systems = None

    exposure = None
    import re
    match = re.search(r"exp_?(\d+)", text)
    if match:
        exposure = int(match.group(1))
    return systems, exposure


def format_coverage_assignment(systems=None, exposure=None):
    if systems is None:
        system_part = "all"
    else:
        system_part = "".join(f"sys{int(i)}" for i in sorted(set(systems))) or "all"
    exposure_part = "all" if exposure is None else f"exp{int(exposure)}"
    return f"{system_part}_{exposure_part}"


def coverage_region_for_line(fit, line, exposure_index=None):
    """Return the matching covering-region id for a line, or ``None``.

    This is the single source of truth used by both the interactive graphics
    model and the non-GUI/calculation path.  Older code duplicated this logic
    and one copy accidentally compared wavelength with ``cf.min/cf.max``
    (the parameter bounds 0..1) instead of ``cf.left/cf.right``.
    """
    if not getattr(fit, "cf_fit", False):
        return None
    center = float(line.l()) * (1.0 + float(line.z))
    matched = None
    for i in range(getattr(fit, "cf_num", 0)):
        name = f"cf_{i}"
        if not hasattr(fit, name):
            continue
        cf = getattr(fit, name)
        systems, exposure = parse_coverage_assignment(getattr(cf, "addinfo", "all"))
        if systems is not None and int(line.sys) not in systems:
            continue
        if exposure is not None and (exposure_index is None or exposure != int(exposure_index)):
            continue
        left = -np.inf if getattr(cf, "left", None) is None else float(cf.left)
        right = np.inf if getattr(cf, "right", None) is None else float(cf.right)
        if left <= center <= right:
            matched = i
    return matched


def selected_fit_wavelength_bounds(spectra):
    """Return min/max wavelength of all currently selected finite fit pixels."""
    lo, hi = np.inf, -np.inf
    for spectrum in spectra:
        try:
            x = np.asarray(spectrum.spec.x(), dtype=float)
            mask = np.asarray(spectrum.fit_mask.x(), dtype=bool)
            if x.size != mask.size:
                x = np.asarray(spectrum.spec.norm.x, dtype=float)
                mask = np.asarray(spectrum.fit_mask.norm.x, dtype=bool)
            good = mask & np.isfinite(x)
            if np.any(good):
                lo = min(lo, float(np.min(x[good])))
                hi = max(hi, float(np.max(x[good])))
        except Exception:
            continue
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else None


def sync_single_coverage_region_to_fit(parent, pad_fraction=2e-6):
    """Make a single LFR span every currently selected fitting window.

    A single covering region represents a *global common coverage law*.  Older
    Spectro created it from the current viewport, so lines in other fitting
    windows silently reverted to full coverage.  When exactly one region is
    present we synchronize its wavelength bounds to the union envelope of all
    selected fit pixels.  Multiple regions are left untouched because they may
    intentionally encode wavelength-dependent coverage.
    """
    fit = getattr(parent, "fit", None)
    spectra = getattr(parent, "s", None)
    if fit is None or spectra is None or not getattr(fit, "cf_fit", False) or getattr(fit, "cf_num", 0) != 1:
        return None
    if not hasattr(fit, "cf_0"):
        return None
    bounds = selected_fit_wavelength_bounds(spectra)
    if bounds is None:
        return None
    left, right = bounds
    pad = max((right - left) * float(pad_fraction), 1e-6)
    cf = fit.cf_0
    cf.left, cf.right = left - pad, right + pad
    try:
        if getattr(parent, "plot", None) is not None and len(parent.plot.pcRegions) == 1:
            region = parent.plot.pcRegions[0]
            region.x1, region.x2 = cf.left, cf.right
            region.redraw()
    except Exception:
        pass
    return cf.left, cf.right


def _line_region_id(line):
    """Return the first valid covering-region id attached to a Spectro line."""
    raw = getattr(line, "cf", -1)
    ids = np.atleast_1d(raw).astype(int)
    ids = ids[ids >= 0]
    return int(ids[0]) if ids.size else None


def assigned_coverage_region(line, fit=None):
    """Return the resolved LFR id attached to ``line``, or ``None``.

    When ``fit`` is supplied, stale region ids without a corresponding
    ``cf_i`` parameter are treated as full coverage, matching the historical
    forward-model behaviour.
    """
    region = _line_region_id(line)
    if region is not None and fit is not None and not hasattr(fit, f"cf_{region}"):
        return None
    return region


def split_line_optical_depths(x, lines, fit=None):
    """Split intrinsic line optical depth into full and LFR channels.

    Returns ``(total_by_system, full_by_system, region_by_system)`` where
    ``region_by_system[region][system]`` is the summed intrinsic optical depth
    of lines assigned to that LFR.  The split uses the already-resolved
    ``line.cf`` assignment, so it exactly follows Spectro's later-LFR-wins
    convention established during ``prepareFit``.

    This helper is shared by the forward model and the QSOdisk exporter to
    prevent the two implementations from drifting apart.
    """
    x = np.asarray(x)
    total_by_system = {}
    full_by_system = {}
    region_by_system = {}
    for line in lines:
        try:
            system_id = int(line.sys)
        except Exception:
            continue
        prof = _tau(line.profile)
        if prof.shape != x.shape:
            continue
        total_by_system.setdefault(system_id, np.zeros_like(x, dtype=float))
        total_by_system[system_id] += prof
        region = assigned_coverage_region(line, fit=fit)
        if region is None:
            full_by_system.setdefault(system_id, np.zeros_like(x, dtype=float))
            full_by_system[system_id] += prof
        else:
            region_by_system.setdefault(int(region), {})
            region_by_system[int(region)].setdefault(system_id, np.zeros_like(x, dtype=float))
            region_by_system[int(region)][system_id] += prof
    return total_by_system, full_by_system, region_by_system


def combine_split_optical_depths(x, total_by_system, full_by_system, region_by_system, fit):
    """Reconstruct Spectro coverage transmission from exported tau channels.

    This function contains no line-selection logic: it consumes the already
    split intrinsic optical-depth arrays returned by
    :func:`split_line_optical_depths`.  It is used by the QSOdisk exporter as
    an internal exactness check and documents the standalone bridge semantics.
    """
    x = np.asarray(x)
    mode = coverage_mode_name(getattr(getattr(fit, "covmode", None), "val", 1))
    if not getattr(fit, "cf_fit", False):
        tau_total = np.zeros_like(x, dtype=float)
        for arr in total_by_system.values():
            tau_total += _tau(arr)
        return np.exp(-tau_total)

    if mode == "powerlaw_sysa_global":
        loga = {
            int(system_id): fit.getValue(f"covasys_{int(system_id)}")
            for system_id in total_by_system
            if hasattr(fit, f"covasys_{int(system_id)}")
        }
        return powerlaw_system_product_with_uncovered(total_by_system, 1.0, loga)

    full_tau = np.zeros_like(x, dtype=float)
    for arr in full_by_system.values():
        full_tau += _tau(arr)
    flux = np.exp(-full_tau)
    for region, tau_by_system in sorted(region_by_system.items()):
        if not hasattr(fit, f"cf_{region}"):
            tau_region_total = np.zeros_like(x, dtype=float)
            for arr in tau_by_system.values():
                tau_region_total += _tau(arr)
            flux *= np.exp(-tau_region_total)
            continue
        cf_default = fit.getValue(f"cf_{region}")
        params = {
            "cf": cf_default,
            "log10_a": fit.getValue(f"cova_{region}") if hasattr(fit, f"cova_{region}") else 0.0,
            "log10_a_by_system": {
                int(system_id): fit.getValue(f"covasys_{int(system_id)}")
                for system_id in tau_by_system
                if hasattr(fit, f"covasys_{int(system_id)}")
            },
            "tau_ratio": fit.getValue(f"covr_{region}") if hasattr(fit, f"covr_{region}") else 0.1,
            "cf1": fit.getValue(f"cov1_{region}") if hasattr(fit, f"cov1_{region}") else cf_default,
            "cf2": fit.getValue(f"cov2_{region}") if hasattr(fit, f"cov2_{region}") else cf_default,
            "cfov_norm": fit.getValue(f"covov_{region}") if hasattr(fit, f"covov_{region}") else 1.0,
        }
        flux *= region_transmission(mode, tau_by_system, params)
    return np.clip(flux, 0.0, 1.0)


def combine_line_profiles(x, lines, fit):
    """Combine Spectro line optical depths using the selected coverage model.

    ``line.profile`` is assumed to be intrinsic optical depth on ``x``;
    ``line.sys`` is the kinematic-system index and ``line.cf`` is the existing
    Spectro covering-region assignment. Lines outside a covering region retain
    ordinary full coverage.

    During a verbose comparison fit, ``fit._coverage_profile_sink`` may point
    to a transient dictionary owned by the current spectrum evaluation.  The
    profiling hooks below are intentionally optional and have no effect on the
    numerical result.
    """
    total_started = time.perf_counter()
    x = np.asarray(x)
    lines = list(lines)
    profile = getattr(fit, "_coverage_profile_sink", None)
    if not lines:
        if profile is not None:
            profile["coverage_total_seconds"] = time.perf_counter() - total_started
            profile["coverage_regions"] = []
        return np.ones_like(x, dtype=float)

    grouping_started = time.perf_counter()
    if not getattr(fit, "cf_fit", False):
        tau_total = np.sum([_tau(line.profile) for line in lines], axis=0)
        result = np.exp(-tau_total)
        if profile is not None:
            profile["coverage_grouping_seconds"] = time.perf_counter() - grouping_started
            profile["coverage_regions"] = []
            profile["coverage_total_seconds"] = time.perf_counter() - total_started
        return result

    mode_value = getattr(getattr(fit, "covmode", None), "val", 1)
    mode = coverage_mode_name(mode_value)
    total_by_system, full_by_system, region_tau = split_line_optical_depths(x, lines, fit=fit)

    if mode == "powerlaw_sysa_global":
        loga = {
            int(system_id): fit.getValue(f"covasys_{int(system_id)}")
            for system_id in total_by_system
            if hasattr(fit, f"covasys_{int(system_id)}")
        }
        missing = [k for k in total_by_system if k not in loga]
        if missing:
            raise ValueError(
                "Global system x-PL is missing covasys parameters for systems "
                + ", ".join(str(k) for k in missing)
            )
        flux = powerlaw_system_product_with_uncovered(
            total_by_system, 1.0, loga, profile=profile
        )
        if profile is not None:
            profile["coverage_grouping_seconds"] = time.perf_counter() - grouping_started
            profile["coverage_regions"] = [{
                "region": "global",
                "mode": mode,
                "systems": list(sorted(total_by_system)),
                "n_points": int(x.size),
            }]
            profile["coverage_total_seconds"] = time.perf_counter() - total_started
        return np.clip(flux, 0.0, 1.0)

    full_tau = np.zeros_like(x, dtype=float)
    for arr in full_by_system.values():
        full_tau += arr
    grouping_seconds = time.perf_counter() - grouping_started
    flux = np.exp(-full_tau)
    region_profiles = []
    for region, tau_by_system in region_tau.items():
        cf_default = fit.getValue(f"cf_{region}")
        params = {
            "cf": cf_default,
            "log10_a": fit.getValue(f"cova_{region}") if hasattr(fit, f"cova_{region}") else 0.0,
            "log10_a_by_system": {
                int(system_id): fit.getValue(f"covasys_{int(system_id)}")
                for system_id in tau_by_system
                if hasattr(fit, f"covasys_{int(system_id)}")
            },
            "tau_ratio": fit.getValue(f"covr_{region}") if hasattr(fit, f"covr_{region}") else 0.1,
            "cf1": fit.getValue(f"cov1_{region}") if hasattr(fit, f"cov1_{region}") else cf_default,
            "cf2": fit.getValue(f"cov2_{region}") if hasattr(fit, f"cov2_{region}") else cf_default,
            "cfov_norm": fit.getValue(f"covov_{region}") if hasattr(fit, f"covov_{region}") else 1.0,
        }
        region_profile = {
            "region": int(region),
            "mode": mode,
            "systems": list(sorted(tau_by_system)),
            "n_points": int(x.size),
        }
        region_started = time.perf_counter()
        flux *= region_transmission(mode, tau_by_system, params, profile=region_profile)
        region_profile["total_seconds"] = time.perf_counter() - region_started
        region_profiles.append(region_profile)

    if profile is not None:
        profile["coverage_grouping_seconds"] = grouping_seconds
        profile["coverage_regions"] = region_profiles
        profile["coverage_total_seconds"] = time.perf_counter() - total_started
    return np.clip(flux, 0.0, 1.0)
