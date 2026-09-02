"""Fit source-coverage models on the same pixels and compare AIC/BIC."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import csv
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .coverage_models import (
    COVERAGE_MODE_LABELS,
    coverage_mode_index,
)


PARAMETER_ATTRIBUTES = (
    "val",
    "min",
    "max",
    "step",
    "vary",
    "fit",
    "fit_w",
    "addinfo",
    "left",
    "right",
    "show",
    "unc",
)


@dataclass
class CoverageFitRecord:
    mode: str
    label: str
    topology: str = "current"
    success: bool = False
    message: str = ""
    chi2: float = math.nan
    ndata: int = 0
    nvarys: int = 0
    dof: int = 0
    redchi: float = math.nan
    aic: float = math.nan
    aicc: float = math.nan
    bic: float = math.nan
    comparable: bool = True
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    curves: list[tuple[np.ndarray, np.ndarray] | None] = field(default_factory=list)

    def criterion(self, name: str) -> float:
        key = name.strip().lower()
        return {
            "chi2": self.chi2,
            "reduced chi2": self.redchi,
            "aic": self.aic,
            "aicc": self.aicc,
            "bic": self.bic,
        }.get(key, self.bic)

    @property
    def usable(self) -> bool:
        return bool(self.parameters) and self.comparable and np.isfinite(self.chi2)


def snapshot_fit_parameters(fit) -> dict[str, dict[str, Any]]:
    """Return a parent-free snapshot of all fit parameters."""
    snapshot: dict[str, dict[str, Any]] = {}
    for name, parameter in fit.pars().items():
        attrs: dict[str, Any] = {}
        for attr in PARAMETER_ATTRIBUTES:
            if hasattr(parameter, attr):
                value = getattr(parameter, attr)
                try:
                    value = deepcopy(value)
                except Exception:
                    pass
                attrs[attr] = value
        snapshot[name] = attrs
    return snapshot


def restore_fit_parameters(fit, snapshot: dict[str, dict[str, Any]]) -> None:
    """Restore a snapshot without changing the model's component structure."""
    covmode = snapshot.get("covmode", {}).get("val")
    if covmode is not None:
        fit.set_coverage_model(covmode)

    for name, attrs in snapshot.items():
        try:
            parameter = fit.getPar(name)
        except Exception:
            if "val" not in attrs:
                continue
            try:
                fit.setValue(name, attrs["val"], "val", check=False)
                parameter = fit.getPar(name)
            except Exception:
                continue

        for attr, value in attrs.items():
            try:
                setattr(parameter, attr, deepcopy(value))
            except Exception:
                setattr(parameter, attr, value)


def snapshot_model_curves(parent) -> list[tuple[np.ndarray, np.ndarray] | None]:
    curves: list[tuple[np.ndarray, np.ndarray] | None] = []
    for spectrum in parent.s:
        try:
            line = spectrum.fit.line.norm
            x = np.asarray(line.x, dtype=float)
            y = np.asarray(line.y, dtype=float)
            if x.size and y.size:
                curves.append((x.copy(), y.copy()))
            else:
                curves.append(None)
        except Exception:
            curves.append(None)
    return curves


def _metrics_from_result(result, residual=None) -> dict[str, float | int]:
    """Calculate comparison metrics from the *displayed final model*.

    ``lmfit`` stores the residual evaluated inside the optimizer.  Spectro then
    copies the optimized parameters back into its own parameter objects and
    recalculates the spectrum.  For model-comparison bookkeeping we therefore
    prefer a residual supplied from ``parent.s.chi()`` after that final
    recalculation.  This guarantees that the statistics and stored model curve
    refer to exactly the same forward model.
    """
    if residual is None:
        residual = getattr(result, "residual", [])
    residual = np.asarray(residual, dtype=float).ravel()
    residual = residual[np.isfinite(residual)]
    ndata = int(residual.size)
    chi2 = float(np.sum(residual**2)) if ndata else math.nan
    nvarys = int(getattr(result, "nvarys", len(getattr(result, "var_names", []) or [])))
    dof = max(ndata - nvarys, 0)
    redchi = chi2 / dof if dof > 0 and np.isfinite(chi2) else math.nan

    if ndata > 0 and np.isfinite(chi2):
        aic = chi2 + 2.0 * nvarys
        bic = chi2 + nvarys * math.log(ndata)
        if ndata > nvarys + 1:
            aicc = aic + 2.0 * nvarys * (nvarys + 1) / (ndata - nvarys - 1)
        else:
            aicc = math.inf
    else:
        aic = aicc = bic = math.nan

    return {
        "chi2": chi2,
        "ndata": ndata,
        "nvarys": nvarys,
        "dof": dof,
        "redchi": redchi,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
    }



def _set_parameter_value(fit, name: str, value: float) -> None:
    """Set a fit parameter if it exists, clipping to its declared bounds."""
    try:
        par = fit.getPar(name)
    except Exception:
        return
    if par is None:
        return
    lo = float(getattr(par, "min", -np.inf))
    hi = float(getattr(par, "max", np.inf))
    value = float(np.clip(value, lo, hi))
    fit.setValue(name, value, "val", check=False)


def _cfov_seed_snapshots(parent, original_snapshot, records):
    """Return robust starting states for the two-component overlap fit.

    Starting with ``Cf1 == Cf2`` and ``Cfov_norm == 1`` places the overlap
    parameterization exactly on both a ``min(Cf1, Cf2)`` cusp and a hard bound.
    A finite-difference least-squares optimizer can then remain on the symmetric
    PPC-like solution.  Seed the overlap model from the already-fitted PPC
    solution, deliberately break that symmetry, and try both nesting
    directions.  Only the starting point changes; the forward model is exactly
    the same ``region_transmission('cfov', ...)`` used by an individual fit.
    """
    ppc = next((record for record in records if record.mode == "ppc" and record.usable), None)
    base = deepcopy(ppc.parameters if ppc is not None else original_snapshot)
    seeds = []

    for direction in (1, -1):
        snap = deepcopy(base)
        snap.setdefault("covmode", {})["val"] = coverage_mode_index("cfov")
        seeds.append((snap, direction))
    return seeds


def _initialize_cfov_from_ppc(parent, direction: int) -> None:
    """Break the symmetric Cf1=Cf2 overlap starting point safely."""
    fit = parent.fit
    for region in range(fit.cf_num):
        try:
            cf = float(fit.getValue(f"cf_{region}"))
        except Exception:
            cf = 0.5
        cf = float(np.clip(cf, 0.05, 0.95))
        low = max(0.02, 0.78 * cf)
        high = min(0.98, 1.12 * cf + 0.04)
        if high <= low + 0.02:
            high = min(0.98, low + 0.08)
        cf1, cf2 = (low, high) if direction > 0 else (high, low)
        _set_parameter_value(fit, f"cov1_{region}", cf1)
        _set_parameter_value(fit, f"cov2_{region}", cf2)
        _set_parameter_value(fit, f"covov_{region}", 0.90)


def _masked_segments(x: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Indices of contiguous fitted wavelength segments.

    Besides False pixels in the fit mask, split on abnormally large wavelength
    jumps.  The latter matters for echelle orders or concatenated spectral
    windows where adjacent array elements can still be far apart in wavelength.
    """
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []

    breaks = np.flatnonzero(np.diff(indices) > 1) + 1

    if indices.size > 1:
        dx = np.diff(x[indices])
        local = dx[(np.diff(indices) == 1) & np.isfinite(dx) & (dx > 0)]
        if local.size:
            typical = float(np.median(local))
            if np.isfinite(typical) and typical > 0:
                gap_breaks = np.flatnonzero(dx > 8.0 * typical) + 1
                if gap_breaks.size:
                    breaks = np.unique(np.concatenate([breaks, gap_breaks]))

    return [segment for segment in np.split(indices, breaks) if segment.size]


def available_comparison_modes(parent) -> list[str]:
    modes = [
        "full",
        "ppc",
        "powerlaw",
        "powerlaw_ppc_sysa",
        "powerlaw_sysa_global",
        "powerlaw_radial",
        "powerlaw_ppc",
        "powerlaw_radial_ppc",
        "two_zone",
    ]
    if len(parent.fit.sys) == 2:
        modes.append("cfov")
    return modes


class CoverageModelSelectionDialog(QDialog):
    """Choose which coverage laws are included in the next comparison."""

    def __init__(self, parent, available_modes: list[str], selected_modes=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.available_modes = list(available_modes)
        previous = set(selected_modes or self.available_modes)
        self.checks = {}

        self.setWindowTitle("Select coverage models")
        self.resize(470, 340)
        layout = QVBoxLayout(self)

        for mode in self.available_modes:
            label = COVERAGE_MODE_LABELS[coverage_mode_index(mode)]
            box = QCheckBox(label)
            box.setChecked(mode in previous)
            self.checks[mode] = box
            layout.addWidget(box)

        presets = QHBoxLayout()
        all_button = QPushButton("Select all")
        none_button = QPushButton("Clear")
        all_button.clicked.connect(lambda: self._set_checked(self.available_modes))
        none_button.clicked.connect(lambda: self._set_checked([]))
        presets.addWidget(all_button)
        presets.addWidget(none_button)
        presets.addStretch(1)
        layout.addLayout(presets)

        topology_row = QHBoxLayout()
        topology_row.addWidget(QLabel('LFR setup:'))
        self.topology = QComboBox()
        self.topology.addItem('common screen', 'common')
        if len(parent.fit.sys) > 1:
            self.topology.addItem('one screen per system', 'by_system')
        self.topology.addItem('keep current LFRs', 'current')
        default_topology = getattr(parent, 'coverageComparisonTopology', None)
        if default_topology is None:
            default_topology = 'by_system' if len(parent.fit.sys) > 1 else 'common'
        index = self.topology.findData(default_topology)
        self.topology.setCurrentIndex(index if index >= 0 else 0)
        self.topology.currentIndexChanged.connect(self._topology_changed)
        topology_row.addWidget(self.topology)
        topology_row.addStretch(1)
        layout.addLayout(topology_row)
        self._topology_changed()

        layout.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        run = QPushButton("Run selected")
        cancel.clicked.connect(self.reject)
        run.clicked.connect(self._accept_if_valid)
        buttons.addWidget(cancel)
        buttons.addWidget(run)
        layout.addLayout(buttons)

    def _set_checked(self, modes):
        selected = set(modes)
        for mode, box in self.checks.items():
            box.setChecked(mode in selected and box.isEnabled())

    def selected_modes(self) -> list[str]:
        return [mode for mode in self.available_modes if self.checks[mode].isChecked()]

    def selected_topology(self) -> str:
        return str(self.topology.currentData())

    def _topology_changed(self):
        by_system = self.selected_topology() == 'by_system'
        box = self.checks.get('cfov')
        if box is not None:
            box.setEnabled(not by_system)
            if by_system:
                box.setChecked(False)
                box.setToolTip('Overlap Cf uses one common LFR containing both systems.')
            else:
                box.setToolTip('')
        for mode, equivalent in [('powerlaw_ppc_sysa', 'Power-law + uncovered (x)'),
                                 ('powerlaw_sysa_global', 'Power-law inhomogeneous (x)')]:
            box = self.checks.get(mode)
            if box is None:
                continue
            box.setEnabled(not by_system)
            if by_system:
                box.setChecked(False)
                box.setToolTip(f'Redundant for one LFR per system; identical to {equivalent}.')
            else:
                box.setToolTip('')

    def _accept_if_valid(self):
        if not self.selected_modes():
            QMessageBox.information(
                self, "Coverage-model comparison",
                "Select at least one coverage model to fit.",
            )
            return
        self.accept()


def select_comparison_modes(parent):
    """Ask the user which applicable models to run; return None on cancel.

    Refuse to open the selector when the fit model contains no absorption
    systems.  Coverage-only fits with tau == 0 are mathematically
    unidentifiable: PPC and both power-law laws all reduce to unit
    transmission, which can otherwise produce a misleading tied BIC result.
    """
    if len(parent.fit.sys) == 0:
        raise ValueError(
            "No absorption systems are present in the current Fit model. "
            "Add or restore at least one system (sys0, sys1, ...) before "
            "running Compare coverage models."
        )
    available = available_comparison_modes(parent)
    previous = getattr(parent, "coverageComparisonSelectedModes", None)
    if previous is not None:
        previous = [mode for mode in previous if mode in available]
    dialog = CoverageModelSelectionDialog(parent, available, previous)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    selected = dialog.selected_modes()
    parent.coverageComparisonSelectedModes = list(selected)
    topology = dialog.selected_topology()
    parent.coverageComparisonTopology = topology
    return selected, topology


def best_record(records: list[CoverageFitRecord], criterion: str = "BIC") -> CoverageFitRecord | None:
    candidates = [
        r for r in records
        if r.usable and r.success and np.isfinite(r.criterion(criterion))
    ]
    if not candidates:
        candidates = [
            r for r in records
            if r.usable and np.isfinite(r.criterion(criterion))
        ]
    return min(candidates, key=lambda r: r.criterion(criterion)) if candidates else None


def apply_record(parent, record: CoverageFitRecord, redraw: bool = True) -> None:
    """Activate a stored fit without rerunning the minimizer."""
    restore_fit_parameters(parent.fit, record.parameters)
    parent.fitType = "regular"
    try:
        parent.options("fitType", "regular")
    except Exception:
        pass
    parent.fit.update(redraw=False)

    missing_curve = False
    for spectrum, curve in zip(parent.s, record.curves):
        if curve is None:
            missing_curve = True
            continue
        x, y = curve
        spectrum.set_fit(x=x, y=y)
        if redraw:
            spectrum.update_fit()
            spectrum.set_gfit()

    if missing_curve:
        parent.s.prepareFit(all=False)
        parent.s.calcFit(recalc=True, redraw=redraw)

    if getattr(parent, "fitModel", None) is not None:
        try:
            parent.fitModel.refresh()
        except Exception:
            pass

    try:
        parent.chiSquare.setText(
            "  coverage model: {0}; chi2 / dof / AICc / BIC = "
            "{1:.2f} / {2:d} / {3:.2f} / {4:.2f}".format(
                record.label, record.chi2, record.dof, record.aicc, record.bic
            )
        )
    except Exception:
        pass

    if redraw:
        try:
            parent.s.calcFitComps(recalc=True)
        except Exception:
            pass
        try:
            parent.s.redraw()
        except Exception:
            pass


def _redshift_drift(snapshot, fit):
    c = 299792.458
    out = []
    for i, system in enumerate(fit.sys):
        name = f'z_{i}'
        if name not in snapshot:
            continue
        z0 = float(snapshot[name].get('val', system.z.val))
        dv = c * (float(system.z.val) - z0) / (1.0 + z0)
        out.append((i, dv))
    return out


def fit_all_coverage_models(parent, modes=None, topology='current') -> list[CoverageFitRecord]:
    """Fit selected applicable coverage models and leave the BIC-preferred fit active."""
    if len(parent.fit.sys) == 0:
        raise ValueError(
            "No absorption systems are present in the current Fit model. "
            "A coverage comparison without any system has tau=0 everywhere "
            "and is not physically meaningful."
        )
    topology = str(topology or 'current')
    if topology != 'current':
        parent.configureCoverageLFRs(topology, reset_vary=True)
    elif not getattr(parent.fit, "cf_fit", False) or parent.fit.cf_num < 1:
        raise ValueError("Create an LFR or choose an automatic LFR setup.")
    if len(parent.fit.list_fit()) == 0:
        raise ValueError("No parameters are selected for fitting.")

    available_modes = available_comparison_modes(parent)
    if modes is None:
        modes = list(available_modes)
    else:
        modes = [mode for mode in modes if mode in available_modes]
    if not modes:
        raise ValueError("No applicable coverage models were selected.")
    if topology == 'by_system' and 'cfov' in modes:
        modes = [mode for mode in modes if mode != 'cfov']
        print('[coverage] overlap Cf skipped: it requires a common LFR containing both systems.', flush=True)
    if not modes:
        raise ValueError('No selected coverage model is compatible with this LFR setup.')

    parent.fit.update(redraw=False)
    parent.s.prepareFit(all=False)
    audit = parent.coverageAssignmentAudit(prepared=True)
    local_modes = {
        'ppc', 'powerlaw', 'powerlaw_ppc', 'powerlaw_ppc_sysa',
        'powerlaw_radial', 'powerlaw_radial_ppc', 'two_zone', 'cfov'
    }
    if audit['unassigned'] and any(mode in local_modes for mode in modes):
        raise ValueError(
            'Some fitted line/system entries are outside the LFR assignment: '
            + ', '.join(audit['unassigned'][:8])
            + '. Choose common/by-system automatic LFRs or fix the manual LFRs.'
        )
    print(f"[coverage] LFR topology={topology}", flush=True)
    for row in audit['topology']:
        print(
            f"[coverage] LFR{row['region']} {row['left']:.3f}-{row['right']:.3f} "
            f"{row['assignment']}",
            flush=True,
        )
    original_snapshot = snapshot_fit_parameters(parent.fit)
    original_curves = snapshot_model_curves(parent)
    original_fit_type = parent.fitType
    original_animate = getattr(parent, "animateFit", False)
    original_option_fit_type = None
    try:
        original_option_fit_type = parent.options("fitType")
    except Exception:
        pass

    progress = QProgressDialog(
        "Preparing coverage-model comparison…", "Cancel", 0, len(modes), parent
    )
    progress.setWindowTitle("Coverage-model comparison")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)

    parent.fitType = "regular"
    parent.animateFit = False
    try:
        parent.options("fitType", "regular")
    except Exception:
        pass

    records: list[CoverageFitRecord] = []
    fatal_error: Exception | None = None
    comparison_started = time.perf_counter()
    print(
        "[coverage] START comparison | "
        f"models={len(modes)} systems={len(parent.fit.sys)} "
        f"fit_parameters={len(parent.fit.list_fit())}",
        flush=True,
    )
    try:
        for index, mode in enumerate(modes):
            if progress.wasCanceled():
                break
            label = COVERAGE_MODE_LABELS[coverage_mode_index(mode)]
            progress.setLabelText(f"Fitting {label} ({index + 1}/{len(modes)})")
            progress.setValue(index)
            QApplication.processEvents()

            record = CoverageFitRecord(mode=mode, label=label, topology=topology)
            model_started = time.perf_counter()
            print(
                f"[coverage] MODEL {index + 1}/{len(modes)} START: {label}",
                flush=True,
            )

            attempts = [(deepcopy(original_snapshot), None)]
            if mode == "cfov":
                attempts = _cfov_seed_snapshots(parent, original_snapshot, records)

            best_attempt = None
            attempt_messages = []
            for attempt_index, (attempt_snapshot, cfov_direction) in enumerate(attempts):
                try:
                    restore_fit_parameters(parent.fit, attempt_snapshot)
                    parent.fit.set_coverage_model(mode)
                    parent.fit.configure_coverage_parameters(reset_vary=True)
                    if mode == "cfov" and cfov_direction is not None:
                        _initialize_cfov_from_ppc(parent, cfov_direction)

                    progress_label = (
                        f"{label} [{index + 1}/{len(modes)}]"
                        if len(attempts) == 1
                        else
                        f"{label} [{index + 1}/{len(modes)}] "
                        f"start {attempt_index + 1}/{len(attempts)}"
                    )
                    result = parent.fitAbs(
                        timer=False,
                        redraw=False,
                        return_result=True,
                        quiet=True,
                        progress_label=progress_label,
                        progress_every=50,
                    )

                    final_residual = np.asarray(parent.s.chi(), dtype=float)
                    metrics = _metrics_from_result(result, residual=final_residual)
                    message = str(getattr(result, "message", ""))
                    if any(name.startswith('FeIIj') for name in parent.fit.list_species()):
                        large = [(i, dv) for i, dv in _redshift_drift(original_snapshot, parent.fit) if abs(dv) > 200.0]
                        if large:
                            warning = 'large z drift: ' + ', '.join(f'sys{i} {dv:+.0f} km/s' for i, dv in large)
                            message = f'{message}; {warning}' if message else warning
                    candidate = {
                        "metrics": metrics,
                        "success": bool(getattr(result, "success", True)),
                        "message": message,
                        "parameters": snapshot_fit_parameters(parent.fit),
                        "curves": snapshot_model_curves(parent),
                        "attempt": attempt_index + 1,
                    }
                    if best_attempt is None or (
                        np.isfinite(metrics["chi2"])
                        and (not np.isfinite(best_attempt["metrics"]["chi2"])
                             or metrics["chi2"] < best_attempt["metrics"]["chi2"])
                    ):
                        best_attempt = candidate
                    attempt_messages.append(
                        f"start {attempt_index + 1}: chi2={metrics['chi2']:.3f}"
                    )
                except Exception as exc:
                    attempt_messages.append(
                        f"start {attempt_index + 1}: {type(exc).__name__}: {exc}"
                    )

            if best_attempt is None:
                record.success = False
                record.message = "; ".join(attempt_messages) or "fit failed"
            else:
                for name, value in best_attempt["metrics"].items():
                    setattr(record, name, value)
                record.success = best_attempt["success"]
                record.message = best_attempt["message"]
                if len(attempts) > 1:
                    detail = (
                        f"overlap multi-start chose {best_attempt['attempt']}/{len(attempts)}; "
                        + ", ".join(attempt_messages)
                    )
                    record.message = f"{record.message}; {detail}" if record.message else detail
                record.parameters = best_attempt["parameters"]
                record.curves = best_attempt["curves"]

                restore_fit_parameters(parent.fit, record.parameters)
                parent.fit.update(redraw=False)
                parent.s.prepareFit(all=False)
                parent.s.calcFit(recalc=True, redraw=False)

            records.append(record)
            model_elapsed = time.perf_counter() - model_started
            if np.isfinite(record.chi2):
                print(
                    f"[coverage] MODEL {index + 1}/{len(modes)} END: {label} | "
                    f"chi2={record.chi2:.6g} redchi={record.redchi:.6g} "
                    f"BIC={record.bic:.6g} elapsed={model_elapsed/60.0:.1f} min",
                    flush=True,
                )
            else:
                print(
                    f"[coverage] MODEL {index + 1}/{len(modes)} END: {label} | "
                    f"FAILED elapsed={model_elapsed/60.0:.1f} min | {record.message}",
                    flush=True,
                )

        progress.setValue(len(modes))
    except Exception as exc:
        fatal_error = exc
    finally:
        progress.close()
        parent.animateFit = original_animate

    if fatal_error is not None:
        restore_fit_parameters(parent.fit, original_snapshot)
        parent.fitType = original_fit_type
        if original_option_fit_type is not None:
            try:
                parent.options("fitType", original_option_fit_type)
            except Exception:
                pass
        raise fatal_error

    counts = [r.ndata for r in records if r.parameters and np.isfinite(r.chi2)]
    if counts:
        reference_ndata = max(set(counts), key=counts.count)
        for record in records:
            if record.parameters and record.ndata != reference_ndata:
                record.comparable = False
                suffix = f"different fitted-pixel count ({record.ndata} vs {reference_ndata})"
                record.message = f"{record.message}; {suffix}" if record.message else suffix

    preferred = best_record(records, "BIC")
    if preferred is not None:
        apply_record(parent, preferred, redraw=True)
    else:
        restore_fit_parameters(parent.fit, original_snapshot)
        for spectrum, curve in zip(parent.s, original_curves):
            if curve is not None:
                spectrum.set_fit(x=curve[0], y=curve[1])
                spectrum.update_fit()
        parent.fitType = original_fit_type
        if original_option_fit_type is not None:
            try:
                parent.options("fitType", original_option_fit_type)
            except Exception:
                pass

    total_elapsed = time.perf_counter() - comparison_started
    preferred_label = preferred.label if preferred is not None else "none"
    print(
        f"[coverage] END comparison | completed={len(records)}/{len(modes)} "
        f"preferred(BIC)={preferred_label} "
        f"elapsed={total_elapsed/60.0:.1f} min",
        flush=True,
    )

    parent.coverageComparisonResults = records
    return records


class CoverageComparisonDialog(QDialog):
    """Results table, model switcher, and comparison plot."""

    CRITERIA = ("BIC", "AICc", "AIC", "chi2", "Reduced chi2")

    def __init__(self, parent, records: list[CoverageFitRecord] | None = None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.records = list(records if records is not None else getattr(parent, "coverageComparisonResults", []))
        self.setWindowTitle("Coverage-model comparison")
        self.resize(1050, 520)

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Preferred-model criterion:"))
        self.criterion = QComboBox()
        self.criterion.addItems(self.CRITERIA)
        self.criterion.setCurrentText("BIC")
        self.criterion.currentTextChanged.connect(self.refresh_table)
        controls.addWidget(self.criterion)

        controls.addWidget(QLabel("Plot exposure:"))
        self.exposure = QComboBox()
        self.exposure.addItems([f"exp {i}" for i in range(len(parent.s))])
        if len(parent.s):
            active_index = getattr(parent.s, "ind", 0)
            active_index = 0 if active_index is None else int(active_index)
            self.exposure.setCurrentIndex(min(len(parent.s) - 1, max(0, active_index)))
        controls.addWidget(self.exposure)

        self.preview = QCheckBox("Preview selected row in main window")
        self.preview.setChecked(True)
        controls.addWidget(self.preview)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.summary = QLabel("")
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            ["Model", "Status", "chi2", "dof", "red. chi2", "k", "AIC", "AICc", "BIC", "Delta", "Message"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.currentCellChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self.apply_selected())
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply selected")
        self.apply_button.clicked.connect(self.apply_selected)
        buttons.addWidget(self.apply_button)

        self.best_button = QPushButton("Apply preferred")
        self.best_button.clicked.connect(self.apply_preferred)
        buttons.addWidget(self.best_button)

        self.plot_button = QPushButton("Plot comparison")
        self.plot_button.clicked.connect(self.plot_comparison)
        buttons.addWidget(self.plot_button)

        self.export_button = QPushButton("Export table…")
        self.export_button.clicked.connect(self.export_table)
        buttons.addWidget(self.export_button)

        buttons.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self.refresh_table()

    def set_records(self, records: list[CoverageFitRecord]) -> None:
        self.records = list(records)
        self.refresh_table()

    def _selected_record(self) -> CoverageFitRecord | None:
        row = self.table.currentRow()
        return self.records[row] if 0 <= row < len(self.records) else None

    def _selection_changed(self, *_args) -> None:
        if self.preview.isChecked():
            record = self._selected_record()
            if record is not None and record.usable:
                apply_record(self.parent_viewer, record, redraw=True)

    def preferred(self) -> CoverageFitRecord | None:
        return best_record(self.records, self.criterion.currentText())

    @staticmethod
    def _format(value: float) -> str:
        return "—" if not np.isfinite(value) else f"{value:.3f}"

    def refresh_table(self) -> None:
        current_mode = self._selected_record().mode if self._selected_record() else None
        criterion = self.criterion.currentText()
        preferred = self.preferred()
        finite_values = [r.criterion(criterion) for r in self.records if r.usable and np.isfinite(r.criterion(criterion))]
        minimum = min(finite_values) if finite_values else math.nan

        self.table.blockSignals(True)
        self.table.setRowCount(len(self.records))
        preferred_row = -1
        selected_row = -1
        for row, record in enumerate(self.records):
            if record.mode == current_mode:
                selected_row = row
            if preferred is record:
                preferred_row = row
            delta = record.criterion(criterion) - minimum if np.isfinite(minimum) and np.isfinite(record.criterion(criterion)) else math.nan
            if not record.comparable and record.parameters:
                status = "different pixels"
            else:
                status = "OK" if record.success else ("usable" if record.usable else "failed")
            values = [
                record.label + ("  [preferred]" if preferred is record else ""),
                status,
                self._format(record.chi2),
                str(record.dof) if record.dof else "—",
                self._format(record.redchi),
                str(record.nvarys),
                self._format(record.aic),
                self._format(record.aicc),
                self._format(record.bic),
                self._format(delta),
                record.message,
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column in range(2, 10):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, column, item)

        row_to_select = selected_row if selected_row >= 0 else preferred_row
        if row_to_select >= 0:
            self.table.selectRow(row_to_select)
        self.table.blockSignals(False)

        if preferred is None:
            self.summary.setText("No usable comparison result.")
        else:
            excluded = sum(1 for record in self.records if not record.comparable)
            note = (
                f" {excluded} result(s) were excluded because their fitted-pixel count differed."
                if excluded else ""
            )
            self.summary.setText(
                f"Preferred by {criterion}: {preferred.label}. "
                "All compared fits use the same masks, pixels, and uncertainties." + note
            )

    def apply_selected(self) -> None:
        record = self._selected_record()
        if record is None or not record.usable:
            QMessageBox.warning(self, "Coverage comparison", "The selected fit is not usable.")
            return
        apply_record(self.parent_viewer, record, redraw=True)

    def apply_preferred(self) -> None:
        record = self.preferred()
        if record is None:
            QMessageBox.warning(self, "Coverage comparison", "No preferred fit is available.")
            return
        apply_record(self.parent_viewer, record, redraw=True)
        self.table.selectRow(self.records.index(record))

    def _spectrum_arrays(self, index: int):
        spectrum = self.parent_viewer.s[index]
        try:
            x = np.asarray(spectrum.spec.norm.x, dtype=float)
            y = np.asarray(spectrum.spec.norm.y, dtype=float)
            err = np.asarray(spectrum.spec.norm.err, dtype=float)
            mask = np.asarray(spectrum.fit_mask.norm.x, dtype=bool)
        except Exception:
            x = np.asarray(spectrum.spec.x(), dtype=float)
            y = np.asarray(spectrum.spec.y(), dtype=float)
            err = np.asarray(spectrum.spec.err(), dtype=float)
            mask = np.asarray(spectrum.fit_mask.x(), dtype=bool)
        if mask.size != x.size:
            mask = np.ones(x.size, dtype=bool)
        mask &= np.isfinite(x) & np.isfinite(y) & np.isfinite(err) & (err > 0)
        return x, y, err, mask

    def plot_comparison(self) -> None:
        if not self.records or len(self.parent_viewer.s) == 0:
            return
        exp_index = max(0, self.exposure.currentIndex())
        x, y, err, mask = self._spectrum_arrays(exp_index)
        if not np.any(mask):
            QMessageBox.warning(self, "Coverage comparison", "The selected exposure has no fitted pixels.")
            return

        preferred = self.preferred()
        segments = _masked_segments(x, mask)
        if not segments:
            QMessageBox.warning(self, "Coverage comparison", "The selected exposure has no fitted segments.")
            return

        fig, (axis, residual_axis) = plt.subplots(
            2, 1, sharex=True, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}
        )

        for segment_index, segment in enumerate(segments):
            axis.plot(
                x[segment], y[segment], ".", markersize=2.5, color="black",
                label="data" if segment_index == 0 else "_nolegend_",
            )

        usable_records = [
            record for record in self.records
            if record.usable and exp_index < len(record.curves)
            and record.curves[exp_index] is not None
        ]
        palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not palette:
            palette = [f"C{i}" for i in range(max(1, len(usable_records)))]
        model_colors = {
            record.mode: palette[i % len(palette)]
            for i, record in enumerate(usable_records)
        }

        plotted = 0
        for record in usable_records:
            curve = record.curves[exp_index]
            model_x, model_y = curve
            label = record.label + (" (preferred)" if record is preferred else "")
            color = model_colors[record.mode]
            drew_record = False
            for segment_index, segment in enumerate(segments):
                model = np.interp(
                    x[segment], model_x, model_y, left=np.nan, right=np.nan
                )
                finite = np.isfinite(model)
                if not np.any(finite):
                    continue
                seg = segment[finite]
                mod = model[finite]
                axis.plot(
                    x[seg], mod, linewidth=1.2, color=color,
                    label=label if not drew_record else "_nolegend_",
                )
                residual_axis.plot(
                    x[seg], (y[seg] - mod) / err[seg], linewidth=0.8,
                    color=color,
                    label=record.label if not drew_record else "_nolegend_",
                )
                drew_record = True
            if drew_record:
                plotted += 1

        if plotted == 0:
            plt.close(fig)
            QMessageBox.warning(self, "Coverage comparison", "No stored model curves are available.")
            return

        axis.set_ylabel("Normalized flux")
        axis.legend(loc="best", fontsize="small", ncol=2)
        axis.set_title(f"Coverage-model comparison: exposure {exp_index}")
        residual_axis.axhline(0.0, linewidth=0.8)
        residual_axis.set_ylabel("Residual / sigma")
        residual_axis.set_xlabel("Wavelength")
        fig.tight_layout()
        plt.show()

    def export_table(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export coverage comparison",
            str(Path(getattr(self.parent_viewer, "work_folder", ".")) / "coverage_model_comparison.csv"),
            "CSV files (*.csv)",
        )
        if not filename:
            return
        with open(filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["mode", "label", "topology", "success", "chi2", "ndata", "nvarys", "dof", "redchi", "AIC", "AICc", "BIC", "message"])
            for record in self.records:
                writer.writerow([
                    record.mode,
                    record.label,
                    record.topology,
                    record.success,
                    record.chi2,
                    record.ndata,
                    record.nvarys,
                    record.dof,
                    record.redchi,
                    record.aic,
                    record.aicc,
                    record.bic,
                    record.message,
                ])

    def closeEvent(self, event) -> None:
        self.parent_viewer.coverageComparison = None
        event.accept()
