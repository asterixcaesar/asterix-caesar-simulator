# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fundación CENER
"""ASTERIx-CAESar | Simplified CSP-CAES techno-economic assessment.

Public website backend for preliminary plant sizing and hourly dispatch studies.
The model uses PyPSA and HiGHS with a rolling-horizon mixed-integer formulation.
It is a screening model, not a detailed plant design or investment forecast.

Main assumptions
----------------
* Hourly electricity prices (USD/MWh) and typical meteorological year DNI
  (W/m2) are supplied in two local, whitespace-delimited text files per country.
  The second column contains values, in chronological order, with 8760 rows
  after two (prices) or five (DNI) header lines. Both series must be aligned.
* The model uses 2025 as a non-leap-year reference calendar, fixed conversion
  factors and isothermal ideal-gas air storage.
* By default, 3 days are optimized and 2 days are committed per rolling window.
* Solar collection during expansion is permitted; compression and expansion
  cannot coincide by default. Turbomachinery can operate at part load.
* TES charging and discharging are mutually exclusive. Air and TES start at
  50% of usable capacity and, by default, both finish at 50%.
* Dispatch maximizes electricity trading margin plus industrial process heat
  (IPH) sales by default, assuming full heat offtake at a fixed price.
  IPH above the adopted correlation limit is not accepted by default.
* Fixed O&M is 1.5% of direct CAPEX, not gross CAPEX. Financial results repeat
  the simulated annual cash flow over the project lifetime in real terms;
  no taxes, financing, degradation or replacements.

The original cost and performance correlations are retained. Their validity
ranges, currency basis and cost-year assumptions require project-team approval.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy_financial as npf
import pandas as pd

if TYPE_CHECKING:
    import pypsa

LOGGER = logging.getLogger(__name__)

# Use NumPy-backed strings where this pandas compatibility option exists.
# This avoids ArrowStringArray errors in some PyPSA/xarray environments.
# The option is deprecated or absent in some pandas releases.
if hasattr(getattr(pd.options, "future", None), "infer_string"):
    pd.options.future.infer_string = False
pd.options.mode.string_storage = "python"

# Model assumptions ----------------------------------------------------------
MODEL_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" if (BASE_DIR / "data").is_dir() else BASE_DIR
PROJECT_CURRENCY = "USD"
REFERENCE_YEAR = 2025
WINDOW_DAYS = 3
COMMIT_DAYS = 2
AIR_TEMPERATURE_K = 303.15
AIR_GAS_CONSTANT_J_KG_K = 287.05
SMALL_SCALE_THRESHOLD_MW = 20.0

P_MIN_BAR = 40.0
P_MAX_SMALL_SCALE_BAR = 150.0
P_MAX_LARGE_SCALE_BAR = 80.0

MW_TO_KGH = 6600.0  # kg/h of air per MW of compressor electricity
KGH_TO_MW = 1.975e-4  # MW of electricity per kg/h of expanded air
CYCLE_EFFICIENCY = 0.50  # electrical output / thermal input
TES_CHARGING_EFFICIENCY = 0.97
TES_STANDING_LOSS_PER_HOUR = 0.001
INITIAL_AIR_SOC_FRACTION = 0.50
INITIAL_TES_SOC_FRACTION = 0.50
ENFORCE_END_OF_YEAR_SOC = True
ALLOW_SOLAR_WHILE_EXPANDING = True
ALLOW_COMPRESSION_WHILE_EXPANDING = False
TURBOMACHINERY_FULL_LOAD_ONLY = False
OPTIMIZATION_CONSIDERING_IPH = True
SELL_IPH = True
ALLOW_HIGHER_IPH = False

IPH_MAX_CORRELATION_POWER_MW = 40.0  # exclusive upper bound [MWth]
IPH_HEAT_PRICE = 50.0  # USD/MWhth
SUBSIDY_PERCENT = 0.0
OTHERS_FRACTION_OF_TOTAL_CAPEX = 0.30
O_AND_M_FRACTION_OF_DIRECT_CAPEX = 0.015
REAL_DISCOUNT_RATE = 0.06
PROJECT_LIFE_YEARS = 30

SOLVER_NAME = "highs"
SOLVER_THREADS = 1
MIP_REL_GAP = 0.01
SOLVER_IO_API = "direct"
# Negligible tie-break cost avoids a completely empty objective in zero-price
# windows and discourages unnecessary solar collection. Excluded from cash flow.
SOLAR_TIE_BREAK_COST = 1e-6  # USD/MWhth


class InvalidDesignError(ValueError):
    """Plant inputs fall outside an adopted assumption or correlation."""


class InputDataError(ValueError):
    """An electricity-price or solar-resource file is missing or invalid."""


class SolverOptimizationError(RuntimeError):
    """The operational optimization did not produce an acceptable solution."""


# Financial and empirical correlations ---------------------------------------
def calculate_financial_metrics(
    total_income: float,
    capex: float,
    operation_maintenance_factor: float = O_AND_M_FRACTION_OF_DIRECT_CAPEX,
    real_discount_rate: float = REAL_DISCOUNT_RATE,
    total_project_life: int = PROJECT_LIFE_YEARS,
    subsidy_percent: float = SUBSIDY_PERCENT,
    *,
    direct_capex: float | None = None,
) -> tuple[float, float | None, float | None]:
    """Return NPV [USD], IRR [fraction], and discounted payback [years].

    CAPEX is gross investment; the subsidy affects only initial investment.
    Fixed O&M is proportional to direct CAPEX (before subsidy). If omitted,
    direct CAPEX is inferred from gross CAPEX and the standard indirect-cost
    fraction. For other cost breakdowns, pass direct_capex explicitly.
    Undefined metrics are None. Positional argument order is unchanged.
    """
    if direct_capex is None:
        direct_capex = capex * (1 - OTHERS_FRACTION_OF_TOTAL_CAPEX)
    if not all(np.isfinite(v) for v in (total_income, capex, direct_capex,
                                        operation_maintenance_factor,
                                        real_discount_rate, subsidy_percent)):
        raise ValueError("Financial inputs must be finite.")
    if capex <= 0 or not 0 < direct_capex <= capex or operation_maintenance_factor < 0:
        raise ValueError("Require 0 < direct CAPEX <= gross CAPEX and a non-negative O&M factor.")
    if real_discount_rate <= -1 or not 0 <= subsidy_percent < 100:
        raise ValueError("Invalid discount rate or subsidy percentage.")
    if not isinstance(total_project_life, (int, np.integer)) or total_project_life <= 0:
        raise ValueError("Project lifetime must be a positive integer.")

    initial_investment = capex * (1 - subsidy_percent / 100)
    annual_cash_flow = total_income - direct_capex * operation_maintenance_factor
    cash_flows = [-initial_investment] + [annual_cash_flow] * total_project_life
    npv = float(npf.npv(real_discount_rate, cash_flows))
    irr = None
    if annual_cash_flow > 0:
        irr_raw = npf.irr(cash_flows)
        if np.isfinite(irr_raw):
            irr = float(irr_raw)

    cumulative = -initial_investment
    discounted_payback = None
    for year in range(1, total_project_life + 1):
        discounted_cf = annual_cash_flow / (1 + real_discount_rate) ** year
        previous = cumulative
        cumulative += discounted_cf
        if cumulative >= 0 and discounted_cf > 0:
            discounted_payback = float(year - 1 - previous / discounted_cf)
            break
    return npv, irr, discounted_payback


def calculate_iph(discharging_power, aperture_area, charging_power):
    """Estimate nominal recoverable process heat in charging and discharging [MWth]."""
    charging_iph_power = 0.5857 * charging_power + 0.3284
    atmospheric_hrsg_iph_power = 0.0000465 * aperture_area - 0.0151629
    discharging_iph_power = (
        0.1108886 * discharging_power - 0.0016426
        if discharging_power <= SMALL_SCALE_THRESHOLD_MW
        else 0.0713415 * discharging_power + 0.0910533
    )
    total_charging_iph_power = charging_iph_power
    total_discharging_iph_power = atmospheric_hrsg_iph_power + discharging_iph_power
    nominal_iph_power = max(total_charging_iph_power, total_discharging_iph_power)
    return total_charging_iph_power, total_discharging_iph_power, nominal_iph_power


def calculate_capex_breakdown(
    discharging_power,
    charging_power,
    tes_hours,
    reservoir_volume,
    aperture_area,
    nominal_iph_power,
    others_fraction,
):
    """Return component, direct, indirect and gross CAPEX in USD.

    The original empirical correlations and their assumed USD basis are
    retained. Verify their currency, cost year and technology ranges.
    All costs use the same currency. Subtotals must not be summed again.
    """
    if not 0 <= others_fraction < 1:
        raise ValueError("others_fraction must be in [0, 1).")
    updated_cepci = 796.3
    cycle_efficiency = CYCLE_EFFICIENCY
    turbine_mass_flow = 1.405 * discharging_power + 0.4226
    compressor_mass_flow = 1.8361 * charging_power + 0.2517
    gas_turbine_power = 0.9 * discharging_power
    bottoming_cycle_power = 0.1 * discharging_power

    high_pressure_turbine = (
        380.46
        * turbine_mass_flow
        * (1 / (0.92 - 0.9033))
        * np.log(40 / 8)
        * (1 + np.exp(0.036 * (273.15 + 550 - 1511)))
        * (updated_cepci / 575.4)
    )
    low_pressure_turbine = (
        380.46
        * turbine_mass_flow
        * (1 / (0.92 - 0.875))
        * np.log(8 / 1.013)
        * (1 + np.exp(0.036 * (273.15 + 750 - 1511)))
        * (updated_cepci / 575.4)
    )
    generator_and_gearbox = (
        20000 * ((gas_turbine_power * 1000) / 100) ** 0.6
    ) * (updated_cepci / 796.3) * 1.4

    if discharging_power <= SMALL_SCALE_THRESHOLD_MW:
        bottoming_cycle = bottoming_cycle_power * 1e6 * (updated_cepci / 567.5)
    else:
        bottoming_cycle = (
            (((bottoming_cycle_power / 0.26) * 1000) * 218.44 + 4000000)
            * 0.67
            * (updated_cepci / 567.3)
            + (8220 * (bottoming_cycle_power * 1000) ** 0.7)
            * (updated_cepci / 603.1)
        )

    compressor_train = (
        4
        * (
            1.051
            * (39.5 * compressor_mass_flow / (0.9 - 0.84))
            * ((80 / 1.013) ** (1 / 4))
            * np.log((80 / 1.013) ** (1 / 4))
        )
        * (updated_cepci / 394.3)
    )

    # Original reservoir technology/cost switch at 20 MWe; not interpolated.
    if discharging_power <= SMALL_SCALE_THRESHOLD_MW:
        air_reservoir = reservoir_volume * 6700 * (updated_cepci / 796.3)
    else:
        air_reservoir = (reservoir_volume * 62 + 37500000) * (updated_cepci / 603.1)

    heat_exchangers = (
        8500000 * (turbine_mass_flow / 30) ** 0.7
        + 4000000 * (compressor_mass_flow / 15) ** 0.7
    ) * (updated_cepci / 796.3)

    thermal_storage = (
        ((discharging_power / cycle_efficiency) * tes_hours * 1000)
        * 20
        * (updated_cepci / 567.5)
    )
    solar_field = aperture_area * 75 * (updated_cepci / 567.5)
    receiver = (0.0016 * aperture_area + 17.376) * 35000 * (updated_cepci / 468.2)
    tower = (21.09 * aperture_area + 1029169.92) * (updated_cepci / 796.3)
    blower = (3.23 * aperture_area + 34784.35) * (updated_cepci / 796.3)
    iph = (
        nominal_iph_power * 1000 * 120
        + 593 * 2000 * max(compressor_mass_flow / 15, turbine_mass_flow / 30)
        + nominal_iph_power * 1000 * 6 * 16.5
    ) * (updated_cepci / 796.3)

    direct_components = {
        "turbine_hpt": high_pressure_turbine,
        "turbine_lpt": low_pressure_turbine,
        "generator_and_gearbox": generator_and_gearbox,
        "bottoming_cycle": bottoming_cycle,
        "compressor_train": compressor_train,
        "air_reservoir": air_reservoir,
        "heat_exchangers": heat_exchangers,
        "thermal_tes": thermal_storage,
        "solar_field": solar_field,
        "receiver": receiver,
        "tower": tower,
        "blower": blower,
        "iph": iph,
    }
    components = {name: float(value) for name, value in direct_components.items()}
    direct_capex = sum(components.values())
    others = direct_capex * others_fraction / (1 - others_fraction)
    return {**components, "others": others, "direct_capex": direct_capex,
            "total_gross": direct_capex + others}


def calculate_capex_large_scale(
    discharging_power, charging_power, tes_hours, reservoir_volume,
    aperture_area, nominal_iph_power, others_fraction,
):
    """Return gross CAPEX [USD]; preserve the original scalar-return interface."""
    return calculate_capex_breakdown(
        discharging_power, charging_power, tes_hours, reservoir_volume,
        aperture_area, nominal_iph_power, others_fraction,
    )["total_gross"]


# Input handling and optimization constraints ---------------------------------
def _component_dim(variable: Any) -> str:
    """Identify the component dimension of a Linopy variable."""
    dims = list(variable.dims)
    non_snapshot_dims = [dim for dim in dims if dim != "snapshot"]
    if "name" in non_snapshot_dims:
        return "name"
    if len(non_snapshot_dims) != 1:
        raise RuntimeError(f"Cannot identify component dimension: {dims!r}")
    return non_snapshot_dims[0]


def _select_component(variable: Any, component_name: str) -> Any:
    return variable.sel({_component_dim(variable): component_name})


def make_extra_functionality(
    final_window: bool,
    terminal_air_soc_kg: float,
    terminal_tes_soc_mwh: float,
    enforce_end_of_year_soc: bool = ENFORCE_END_OF_YEAR_SOC,
    *,
    allow_solar_while_expanding: bool = ALLOW_SOLAR_WHILE_EXPANDING,
    allow_compression_while_expanding: bool = ALLOW_COMPRESSION_WHILE_EXPANDING,
    turbomachinery_full_load_only: bool = TURBOMACHINERY_FULL_LOAD_ONLY,
) -> Callable[[pypsa.Network, pd.Index], None]:
    """Build operating-mode and terminal storage constraints for one window."""

    def extra_functionality(n, snapshots):
        m = n.model
        snapshot_coord = pd.Index(snapshots, name="snapshot")
        p_link = m.variables["Link-p"]
        p_expander = _select_component(p_link, "Expander_Turbine")
        p_compressor = _select_component(p_link, "Compressor")
        exp_nom = float(n.links.at["Expander_Turbine", "p_nom"])
        comp_nom = float(n.links.at["Compressor", "p_nom"])

        # Add a binary expander status only when needed by the selected modes.
        needs_expander_status = (turbomachinery_full_load_only
                                 or not allow_compression_while_expanding
                                 or not allow_solar_while_expanding)
        if needs_expander_status:
            expander_active = m.add_variables(
                coords=[snapshot_coord], name="expander_active", binary=True
            )
            if turbomachinery_full_load_only:
                m.add_constraints(p_expander == expander_active * exp_nom,
                                  name="expander_full_load_only")
            else:
                m.add_constraints(p_expander <= expander_active * exp_nom,
                                  name="expander_only_when_active")

        if turbomachinery_full_load_only:
            compressor_active = m.add_variables(
                coords=[snapshot_coord], name="compressor_active", binary=True
            )
            m.add_constraints(p_compressor == compressor_active * comp_nom,
                              name="compressor_full_load_only")

        if not allow_compression_while_expanding:
            if turbomachinery_full_load_only:
                m.add_constraints(compressor_active + expander_active <= 1,
                                  name="no_simultaneous_compression_expansion")
            else:
                m.add_constraints(p_compressor <= (1 - expander_active) * comp_nom,
                                  name="compressor_only_when_expander_inactive")

        # TES charging and discharging are always mutually exclusive.
        tes_charge = _select_component(m.variables["StorageUnit-p_store"], "TES_Battery")
        tes_discharge = _select_component(m.variables["StorageUnit-p_dispatch"], "TES_Battery")
        tes_nom = float(n.storage_units.at["TES_Battery", "p_nom"])
        if not allow_solar_while_expanding:
            p_solar = _select_component(m.variables["Generator-p"], "CSP_Source")
            solar_nom = float(n.generators.at["CSP_Source", "p_nom"])
            m.add_constraints(p_solar <= (1 - expander_active) * solar_nom,
                              name="solar_only_when_expander_inactive")
            tes_discharge_mode = expander_active
        else:
            # With direct solar heat, the TES may charge or discharge during
            # expansion, depending on the instantaneous solar/thermal balance.
            tes_discharge_mode = m.add_variables(
                coords=[snapshot_coord], name="tes_discharge_mode", binary=True
            )
        m.add_constraints(tes_charge <= (1 - tes_discharge_mode) * tes_nom,
                          name="tes_charge_mode")
        m.add_constraints(tes_discharge <= tes_discharge_mode * tes_nom,
                          name="tes_discharge_mode_limit")

        if final_window and enforce_end_of_year_soc:
            final_snapshot = snapshots[-1]
            air_e = _select_component(m.variables["Store-e"], "Air_Reservoir").sel(
                snapshot=final_snapshot
            )
            tes_soc = _select_component(
                m.variables["StorageUnit-state_of_charge"], "TES_Battery"
            ).sel(snapshot=final_snapshot)
            m.add_constraints(air_e == terminal_air_soc_kg, name="end_of_year_air_soc")
            m.add_constraints(tes_soc == terminal_tes_soc_mwh, name="end_of_year_tes_soc")

    return extra_functionality


def _validate_inputs(power_rating, TES_hours, reservoir_volume, aperture_area, charging_power):
    for name, value in {
        "power_rating": power_rating,
        "TES_hours": TES_hours,
        "reservoir_volume": reservoir_volume,
        "aperture_area": aperture_area,
        "charging_power": charging_power,
    }.items():
        if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) or
            not np.isfinite(value) or value <= 0):
            raise InvalidDesignError(f"{name} must be a positive, finite number; got {value!r}.")


def _resolve_input_files(country: str, data_dir: str | Path | None) -> tuple[Path, Path]:
    """Resolve local country files without accepting user-supplied paths."""
    if (not isinstance(country, str) or not country.strip() or
        country in {".", ".."} or any(c in country for c in ("/", "\\", ":", "\x00"))):
        raise InputDataError("Country must be a supported local filename identifier.")
    root = Path(data_dir).resolve() if data_dir is not None else DATA_DIR.resolve()
    # Supplied files use lowercase first letters after TMY, even on Linux.
    solar_name = f"TMY{country[0].lower()}{country[1:]}.txt"
    solar_path = root / solar_name
    if not solar_path.is_file():
        solar_path = root / f"TMY{country}.txt"  # Legacy spelling
    paths = ((root / f"{country}.txt").resolve(), solar_path.resolve())
    if any(path.parent != root for path in paths):
        raise InputDataError("Country data must be located in the configured data directory.")
    return paths


def _load_input_column(path: Path, skiprows: int, required_length: int, label: str) -> np.ndarray:
    try:
        data = np.loadtxt(path, skiprows=skiprows, ndmin=2)
    except (OSError, ValueError) as exc:
        raise InputDataError(f"{label}: cannot read '{path.name}'.") from exc
    if data.shape[1] < 2:
        raise InputDataError(f"{label}: '{path.name}' must contain at least two columns.")
    values = np.asarray(data[:, 1], dtype=float)
    if len(values) != required_length:
        raise InputDataError(
            f"{label}: '{path.name}' contains {len(values)} rows; expected exactly {required_length}."
        )
    if not np.all(np.isfinite(values)):
        raise InputDataError(f"{label}: '{path.name}' contains NaN or infinite values.")
    if not np.array_equal(data[:, 0], np.arange(required_length) * 3600):
        raise InputDataError(f"{label}: '{path.name}' must have consecutive hourly timestamps starting at zero seconds.")
    return values


def _rolling_windows(total_hours: int, window: int, step: int):
    """Yield (start, end, commit_end), merging the short final remainder.

    By default the horizon is 3 days with 2 days committed. Merging a short
    final tail gives the terminal inventory constraint more time to be met.
    It does not guarantee feasibility for every possible design.
    """
    if total_hours <= 0 or not 0 < step <= window:
        raise ValueError("Invalid rolling-horizon configuration.")
    start = 0
    while start < total_hours:
        remaining = total_hours - start
        if remaining <= max(window, 2 * step - 1):
            yield start, total_hours, total_hours
            break
        yield start, min(start + window, total_hours), start + step
        start += step


def _reference_snapshots() -> pd.DatetimeIndex:
    """Return the fixed hourly reference calendar."""
    return pd.date_range(f"{REFERENCE_YEAR}-01-01", f"{REFERENCE_YEAR + 1}-01-01",
                         inclusive="left", freq="h")


def _nonnegative(series: pd.Series, label: str, tolerance: float = 1e-5) -> pd.Series:
    """Reject invalid signs and clip only negligible solver residuals."""
    if not np.isfinite(series.to_numpy()).all() or (series < -tolerance).any():
        raise SolverOptimizationError(f"Invalid optimized values for {label}.")
    return series.clip(lower=0.0)


# Public simulation interface -------------------------------------------------
def simulate_plant(
    power_rating: float,
    TES_hours: float,
    reservoir_volume: float,
    aperture_area: float,
    country: str,
    charging_power: float,
    *,
    data_dir: str | Path | None = None,
    optimize_iph_revenue: bool = OPTIMIZATION_CONSIDERING_IPH,
    enforce_end_of_year_soc: bool = ENFORCE_END_OF_YEAR_SOC,
    solver_time_limit_seconds: float | None = None,
    overall_time_limit_seconds: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    allow_solar_while_expanding: bool = ALLOW_SOLAR_WHILE_EXPANDING,
    allow_compression_while_expanding: bool = ALLOW_COMPRESSION_WHILE_EXPANDING,
    turbomachinery_full_load_only: bool = TURBOMACHINERY_FULL_LOAD_ONLY,
    allow_higher_iph: bool = ALLOW_HIGHER_IPH,
    sell_iph: bool = SELL_IPH,
) -> dict[str, Any]:
    """Simulate one representative year and return financial and dispatch KPIs.

    Parameters
    ----------
    power_rating : float
        Maximum electrical output of the expansion train [MWe].
    TES_hours : float
        Nominal TES discharge duration at rated electrical output [h].
    reservoir_volume : float
        Physical compressed-air reservoir volume [m3].
    aperture_area : float
        Solar-field aperture area [m2].
    country : str
        Local dataset identifier (e.g. Spain, if Spain.txt and TMYSpain.txt exist).
    charging_power : float
        Rated electrical input of the compression train [MWe].
    data_dir : str or Path, optional
        Trusted directory containing both data files; defaults to data/ beside the module.
    optimize_iph_revenue : bool, optional
        If True (default) and sell_iph is True, optimize electricity trading
        and IPH sales jointly; otherwise optimize electricity trading only.
        If IPH is sold but excluded from dispatch, value it in post-processing.
    enforce_end_of_year_soc : bool, optional
        Restore both storage inventories to their initial values (default True).
    solver_time_limit_seconds : float, optional
        Per-window solver limit. A non-optimal termination is reported as an error.
    overall_time_limit_seconds : float, optional
        Cooperative annual execution budget, checked around solver calls.
        This is not a hard process timeout; model construction can overrun it.
    progress_callback : callable, optional
        Receives the number of committed snapshots and total snapshots.
    allow_solar_while_expanding : bool, optional
        Permit direct solar heat during expansion (default True).
    allow_compression_while_expanding : bool, optional
        Permit concurrent compression and expansion (default False). Electricity
        is valued at the same import/export price; IPH capacity is sized for both.
    turbomachinery_full_load_only : bool, optional
        Force ON/OFF operation at rated power (default False). Exact terminal
        inventories may be infeasible with discrete full-load air-mass steps.
    allow_higher_iph : bool, optional
        Allow extrapolation at or above the 40 MWth IPH limit (default False).
        This flag does not switch heat sales on or off.
    sell_iph : bool, optional
        Include IPH sales in cash flow (default True), assuming full offtake.
        False removes heat revenue, not the installed IPH equipment CAPEX.

    Returns
    -------
    dict
        Financial metrics and hourly dispatch DataFrame. Undefined IRR, discounted
        payback and electricity production/consumption ratio are None, not zero.
        Legacy output keys are retained for compatibility with the existing UI.
    """
    _validate_inputs(power_rating, TES_hours, reservoir_volume, aperture_area, charging_power)
    if solver_time_limit_seconds is not None and (
        not np.isfinite(solver_time_limit_seconds) or solver_time_limit_seconds <= 0
    ):
        raise ValueError("Solver time limit must be positive and finite.")

    if overall_time_limit_seconds is not None and (
        not np.isfinite(overall_time_limit_seconds) or overall_time_limit_seconds <= 0
    ):
        raise ValueError("Overall time limit must be positive and finite.")
    started = time.monotonic()

    settings = {
        "optimize_iph_revenue": optimize_iph_revenue,
        "enforce_end_of_year_soc": enforce_end_of_year_soc,
        "allow_solar_while_expanding": allow_solar_while_expanding,
        "allow_compression_while_expanding": allow_compression_while_expanding,
        "turbomachinery_full_load_only": turbomachinery_full_load_only,
        "allow_higher_iph": allow_higher_iph,
        "sell_iph": sell_iph,
    }
    for name, value in settings.items():
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean.")
    # Heat revenue can only enter the optimization objective if it is sold.
    optimize_iph_revenue = bool(optimize_iph_revenue and sell_iph)
    warnings = []
    if not enforce_end_of_year_soc:
        warnings.append("Annual storage inventories are not conserved; results may include net use of initial stock.")

    charging_iph, discharging_iph, nominal_iph = calculate_iph(
        power_rating, aperture_area, charging_power
    )
    if allow_compression_while_expanding:
        # Conservative installed heat-recovery capacity for simultaneous modes.
        # The original correlation only sizes max(charging, discharging).
        nominal_iph = charging_iph + discharging_iph
        warnings.append("Simultaneous compression/expansion uses summed nominal IPH sizing; verify equipment design.")
    if not all(np.isfinite(v) and v >= 0 for v in (charging_iph, discharging_iph, nominal_iph)):
        raise InvalidDesignError("IPH correlations must predict finite, non-negative thermal powers.")
    if nominal_iph >= IPH_MAX_CORRELATION_POWER_MW:
        if not allow_higher_iph:
            raise InvalidDesignError(
                f"Nominal IPH power ({nominal_iph:.2f} MWth) must be below "
                f"{IPH_MAX_CORRELATION_POWER_MW:g} MWth. Adjust the plant ratings "
                "and/or solar-field aperture area."
            )
        warnings.append("Nominal IPH is at or above the adopted 40 MWth limit; correlations are extrapolated.")

    solar_efficiency = -0.02026 * np.log(aperture_area) + 0.79844
    if not 0 < solar_efficiency <= 1:
        raise InvalidDesignError(f"Solar-to-thermal efficiency is outside (0, 1]: {solar_efficiency:.4f}.")

    # Usable air mass, not the total mass including the minimum-pressure cushion.
    p_max_bar = P_MAX_SMALL_SCALE_BAR if power_rating <= SMALL_SCALE_THRESHOLD_MW else P_MAX_LARGE_SCALE_BAR
    air_mass_capacity = ((p_max_bar - P_MIN_BAR) * 1e5 * reservoir_volume /
                         (AIR_GAS_CONSTANT_J_KG_K * AIR_TEMPERATURE_K))
    if not np.isfinite(air_mass_capacity) or air_mass_capacity <= 0:
        raise InvalidDesignError("Calculated usable air mass must be positive and finite.")

    snapshots = _reference_snapshots()
    dt_hours = (snapshots[1] - snapshots[0]).total_seconds() / 3600
    electricity_path, solar_path = _resolve_input_files(country, data_dir)
    prices = _load_input_column(electricity_path, 2, len(snapshots), "Electricity prices")
    solar = _load_input_column(solar_path, 5, len(snapshots), "Solar resource")
    if np.any(solar < 0):
        warning = f"{int(np.sum(solar < 0))} negative irradiance values were clipped to zero."
        LOGGER.warning(warning)
        warnings.append(warning)
        solar = np.clip(solar, 0, None)
    price_series = pd.Series(prices, index=snapshots, dtype=float, name="Price")
    available_solar = pd.Series(solar * aperture_area * solar_efficiency / 1e6,
                                index=snapshots, dtype=float)
    solar_peak = float(available_solar.max())
    if not np.isfinite(solar_peak) or solar_peak <= 0:
        raise InvalidDesignError("Available solar thermal power must be positive and finite.")

    thermal_cycle_input = power_rating / CYCLE_EFFICIENCY
    tes_capacity = TES_hours * thermal_cycle_input
    tes_power = max(thermal_cycle_input, solar_peak)
    if not np.isfinite(tes_capacity) or tes_capacity <= 0:
        raise InvalidDesignError("Calculated TES capacity must be positive and finite.")
    # IPH heat per unit of link input. PyPSA link marginal cost is defined per
    # unit of bus0 input: MW for the compressor and kg/h for the expander.
    compressor_iph_credit = IPH_HEAT_PRICE * charging_iph / charging_power
    expander_iph_credit = IPH_HEAT_PRICE * discharging_iph / power_rating * KGH_TO_MW

    # The supplied Windows environment uses PyPSA 1.2.4. Revalidate the
    # first-hour TES standing-loss convention when changing versions.
    import pypsa

    logging.getLogger("pypsa").setLevel(logging.WARNING)
    logging.getLogger("linopy").setLevel(logging.WARNING)

    # PyPSA network. The air bus uses kg/h and the air store uses kg, rather
    # than PyPSA's conventional MW and MWh. Link factors implement conversion.
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.snapshot_weightings.loc[:, :] = dt_hours
    n.add("Carrier", "AC")
    n.add("Carrier", "compressed_air")
    n.add("Carrier", "thermal")
    n.add("Bus", "Grid", carrier="AC", unit="MW")
    n.add("Bus", "Air_Bus", carrier="compressed_air", unit="kg/h")
    n.add("Bus", "Thermal_Bus", carrier="thermal", unit="MWth")

    # Market dispatch is positive for imports and negative for exports.
    market_capacity = max(power_rating, charging_power)
    n.add("Generator", "Market", bus="Grid", carrier="AC", p_nom=market_capacity,
          p_min_pu=-power_rating / market_capacity,
          p_max_pu=charging_power / market_capacity, marginal_cost=price_series)
    n.add("Generator", "CSP_Source", bus="Thermal_Bus", carrier="thermal",
          p_nom=solar_peak, p_max_pu=available_solar / solar_peak,
          marginal_cost=SOLAR_TIE_BREAK_COST)
    n.add("StorageUnit", "TES_Battery", bus="Thermal_Bus", carrier="thermal",
          p_nom=tes_power, max_hours=tes_capacity / tes_power,
          efficiency_store=TES_CHARGING_EFFICIENCY, efficiency_dispatch=1.0,
          standing_loss=TES_STANDING_LOSS_PER_HOUR, cyclic_state_of_charge=False)
    n.add("Store", "Air_Reservoir", bus="Air_Bus", carrier="compressed_air",
          e_nom=air_mass_capacity, e_min_pu=0.0, e_max_pu=1.0, e_cyclic=False)
    n.add("Link", "Compressor", bus0="Grid", bus1="Air_Bus", bus2="", carrier="compressed_air",
          p_nom=charging_power, efficiency=MW_TO_KGH, efficiency2=0.0, p_min_pu=0.0, p_max_pu=1.0,
          marginal_cost=-compressor_iph_credit if optimize_iph_revenue else 0.0)
    n.add("Link", "Expander_Turbine", bus0="Air_Bus", bus1="Grid", bus2="Thermal_Bus",
          carrier="compressed_air", p_nom=power_rating / KGH_TO_MW,
          efficiency=KGH_TO_MW, efficiency2=-KGH_TO_MW / CYCLE_EFFICIENCY,
          p_min_pu=0.0, p_max_pu=1.0,
          marginal_cost=-expander_iph_credit if optimize_iph_revenue else 0.0)

    # Rolling-horizon dispatch ------------------------------------------------
    initial_air = air_mass_capacity * INITIAL_AIR_SOC_FRACTION
    initial_tes = tes_capacity * INITIAL_TES_SOC_FRACTION
    current_air, current_tes = initial_air, initial_tes
    columns = ["P_Disch", "P_Charge_Positive", "Air_kg", "TES_MWhth",
               "Solar_Captured_MWth", "TES_Charge_MWth", "TES_Discharge_MWth"]
    dispatch = pd.DataFrame(np.nan, index=snapshots, columns=columns)
    solver_options = {"threads": SOLVER_THREADS, "mip_rel_gap": MIP_REL_GAP, "output_flag": False}
    if solver_time_limit_seconds is not None:
        solver_options["time_limit"] = float(solver_time_limit_seconds)
    status = condition = None
    window_count = 0

    for start, end, commit_end in _rolling_windows(len(snapshots), 24 * WINDOW_DAYS, 24 * COMMIT_DAYS):
        window_snaps = snapshots[start:end]
        commit_snaps = snapshots[start:commit_end]
        final_window = commit_end == len(snapshots)
        n.stores.at["Air_Reservoir", "e_initial"] = current_air
        # PyPSA 1.0.3/1.1.x inserts the non-cyclic initial SOC directly into
        # its first balance equation, without applying first-hour standing loss.
        # Pre-decay the carried inventory so all 8760 hours include the loss.
        # Keep the dependency version pinned; revalidate this on PyPSA upgrades.
        n.storage_units.at["TES_Battery", "state_of_charge_initial"] = (
            current_tes * (1 - TES_STANDING_LOSS_PER_HOUR) ** dt_hours
        )
        extra_functionality = make_extra_functionality(
            final_window, initial_air, initial_tes, enforce_end_of_year_soc,
            allow_solar_while_expanding=allow_solar_while_expanding,
            allow_compression_while_expanding=allow_compression_while_expanding,
            turbomachinery_full_load_only=turbomachinery_full_load_only,
        )
        if overall_time_limit_seconds is not None:
            remaining = overall_time_limit_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise SolverOptimizationError("Annual simulation time budget exceeded.")
            solver_options["time_limit"] = min(remaining, solver_time_limit_seconds or remaining)
        try:
            status, condition = n.optimize(
                window_snaps, solver_name=SOLVER_NAME,
                extra_functionality=extra_functionality, solver_options=solver_options,
                io_api=SOLVER_IO_API, log_to_console=False,
                include_objective_constant=False,
            )
        except Exception as exc:
            raise SolverOptimizationError(
                f"Optimization failed for {window_snaps[0]} to {window_snaps[-1]}."
            ) from exc
        if status != "ok" or condition != "optimal":
            raise SolverOptimizationError(
                f"PyPSA/HiGHS did not reach an optimal termination (status={status!r}, "
                f"condition={condition!r}) for {window_snaps[0]} to {window_snaps[-1]}. "
                "Check plant sizing, terminal storage targets and solver limits."
            )
        window_count += 1

        dispatch.loc[commit_snaps, "P_Disch"] = _nonnegative(
            -n.links_t.p1["Expander_Turbine"].loc[commit_snaps], "expander electrical output")
        dispatch.loc[commit_snaps, "P_Charge_Positive"] = _nonnegative(
            n.links_t.p0["Compressor"].loc[commit_snaps], "compressor electrical input")
        dispatch.loc[commit_snaps, "Air_kg"] = n.stores_t.e["Air_Reservoir"].loc[commit_snaps]
        dispatch.loc[commit_snaps, "TES_MWhth"] = n.storage_units_t.state_of_charge["TES_Battery"].loc[commit_snaps]
        dispatch.loc[commit_snaps, "Solar_Captured_MWth"] = _nonnegative(
            n.generators_t.p["CSP_Source"].loc[commit_snaps], "collected solar heat")
        dispatch.loc[commit_snaps, "TES_Charge_MWth"] = _nonnegative(
            n.storage_units_t.p_store["TES_Battery"].loc[commit_snaps], "TES charging")
        dispatch.loc[commit_snaps, "TES_Discharge_MWth"] = _nonnegative(
            n.storage_units_t.p_dispatch["TES_Battery"].loc[commit_snaps], "TES discharging")
        current_air = float(dispatch.loc[commit_snaps[-1], "Air_kg"])
        current_tes = float(dispatch.loc[commit_snaps[-1], "TES_MWhth"])
        if overall_time_limit_seconds is not None and time.monotonic() - started > overall_time_limit_seconds:
            raise SolverOptimizationError("Annual simulation time budget exceeded.")
        if progress_callback is not None:
            progress_callback(commit_end, len(snapshots))

    if not np.isfinite(dispatch.to_numpy()).all():
        raise SolverOptimizationError("The annual dispatch contains missing or non-finite results.")
    _check_dispatch(dispatch, available_solar, power_rating, charging_power,
                    air_mass_capacity, tes_capacity, initial_air, initial_tes,
                    dt_hours, enforce_end_of_year_soc,
                    allow_solar_while_expanding=allow_solar_while_expanding,
                    allow_compression_while_expanding=allow_compression_while_expanding,
                    turbomachinery_full_load_only=turbomachinery_full_load_only,
                    tes_power=tes_power)

    # Annual energy and financial assessment ----------------------------------
    capex_breakdown = calculate_capex_breakdown(
        power_rating, charging_power, TES_hours, reservoir_volume, aperture_area,
        nominal_iph, OTHERS_FRACTION_OF_TOTAL_CAPEX,
    )
    gross_capex = capex_breakdown["total_gross"]
    direct_capex = capex_breakdown["direct_capex"]
    if not np.isfinite(gross_capex) or gross_capex <= 0:
        raise InvalidDesignError("Calculated gross CAPEX must be positive and finite.")
    net_investment = gross_capex * (1 - SUBSIDY_PERCENT / 100)
    p_ch = dispatch["P_Charge_Positive"]
    p_dis = dispatch["P_Disch"]
    energy_charged = float(p_ch.sum() * dt_hours)
    energy_discharged = float(p_dis.sum() * dt_hours)
    solar_captured = float(dispatch["Solar_Captured_MWth"].sum() * dt_hours)
    solar_available = float(available_solar.sum() * dt_hours)
    hourly_iph = charging_iph * p_ch / charging_power + discharging_iph * p_dis / power_rating
    iph_energy = float(hourly_iph.sum() * dt_hours)
    electricity_revenue = float((p_dis * price_series).sum() * dt_hours)
    electricity_cost = float((p_ch * price_series).sum() * dt_hours)
    iph_revenue = iph_energy * IPH_HEAT_PRICE if sell_iph else 0.0
    operating_margin_electricity = electricity_revenue - electricity_cost
    operating_margin = operating_margin_electricity + iph_revenue
    annual_om = direct_capex * O_AND_M_FRACTION_OF_DIRECT_CAPEX
    annual_cash_flow = operating_margin - annual_om
    npv, irr, dpp = calculate_financial_metrics(
        operating_margin, gross_capex, subsidy_percent=SUBSIDY_PERCENT,
        direct_capex=direct_capex,
    )
    # Diagnostic: remove IPH revenue but retain the same dispatch and CAPEX.
    npv_electricity, irr_electricity, dpp_electricity = calculate_financial_metrics(
        operating_margin_electricity, gross_capex, subsidy_percent=SUBSIDY_PERCENT,
        direct_capex=direct_capex,
    )
    average_selling_price = (electricity_revenue / energy_discharged
                             if energy_discharged > 1e-9 else None)
    average_purchase_price = (electricity_cost / energy_charged
                              if energy_charged > 1e-9 else None)
    # External solar input means this ratio is not a thermodynamic round-trip
    # efficiency. With equal initial/final air inventories, constant conversion
    # factors and no air loss, it is fixed by MW_TO_KGH * KGH_TO_MW (130.35%).
    production_ratio = (100 * energy_discharged / energy_charged
                        if energy_charged > 1e-9 else None)
    capacity_factor = 100 * energy_discharged / (power_rating * len(snapshots) * dt_hours)
    df_results = pd.DataFrame({
        "Price": price_series,
        "P_Disch": p_dis,
        "P_Ch": -p_ch,  # Legacy chart convention: charging power is negative.
        "SoC_Thermal_%": 100 * dispatch["TES_MWhth"] / tes_capacity,
        "SoC_Air_%": 100 * dispatch["Air_kg"] / air_mass_capacity,
        "Solar_Captured_MWth": dispatch["Solar_Captured_MWth"],
        "TES_Charge_MWth": dispatch["TES_Charge_MWth"],
        "TES_Discharge_MWth": dispatch["TES_Discharge_MWth"],
        "IPH_MWth": hourly_iph,
        "Solar_Available_MWth": available_solar,
    })
    return {
        "model_version": MODEL_VERSION,
        "currency": PROJECT_CURRENCY,
        "status": (status, condition),
        "solver_status": status,
        "termination_condition": condition,
        "optimization_windows": window_count,
        "dispatch_objective": "electricity_and_iph" if optimize_iph_revenue else "electricity_only",
        "end_of_year_soc_enforced": enforce_end_of_year_soc,
        "operation_settings": {**{key: bool(value) for key, value in settings.items()},
                               "optimize_iph_revenue": optimize_iph_revenue,
                               "window_days": WINDOW_DAYS, "commit_days": COMMIT_DAYS},
        "warnings": warnings,
        "capex_musd": gross_capex / 1e6,
        "direct_capex_musd": direct_capex / 1e6,
        "capex_breakdown_usd": capex_breakdown,
        "net_initial_investment_musd": net_investment / 1e6,
        "npv_musd": npv / 1e6,
        "irr_pct": None if irr is None else 100 * irr,
        "dpp_years": dpp,
        "npv_electricity_only_musd": npv_electricity / 1e6,
        "irr_electricity_only_pct": None if irr_electricity is None else 100 * irr_electricity,
        "dpp_electricity_only_years": dpp_electricity,
        "profit_usd": operating_margin,  # Legacy key: before fixed O&M.
        "operating_margin_usd": operating_margin,
        "operating_margin_electricity_usd": operating_margin_electricity,
        "annual_om_cost_usd": annual_om,
        "om_basis": "direct_capex",
        "annual_cash_flow_usd": annual_cash_flow,
        "erte_pct": production_ratio,  # Legacy alias; not thermodynamic RTE.
        "electricity_production_to_consumption_pct": production_ratio,
        "closed_air_inventory_ratio_pct": 100 * MW_TO_KGH * KGH_TO_MW,
        "cf_pct": capacity_factor,
        "electricity_revenue_usd": electricity_revenue,
        "charging_cost_usd": electricity_cost,
        "average_selling_price_usd_mwh": average_selling_price,
        "average_purchase_price_usd_mwh": average_purchase_price,
        "iph_revenue_usd": iph_revenue,
        "iph_energy_mwhth": iph_energy,
        "iph_nominal_power_mwth": nominal_iph,
        "energy_charged_mwh": energy_charged,
        "energy_discharged_mwh": energy_discharged,
        "solar_captured_mwhth": solar_captured,
        "solar_available_mwhth": solar_available,
        "tes_capacity_mwhth": tes_capacity,
        "air_usable_mass_kg": air_mass_capacity,
        "initial_tes_soc_pct": INITIAL_TES_SOC_FRACTION * 100,
        "final_tes_soc_pct": float(df_results["SoC_Thermal_%"].iloc[-1]),
        "initial_air_soc_pct": INITIAL_AIR_SOC_FRACTION * 100,
        "final_air_soc_pct": float(df_results["SoC_Air_%"].iloc[-1]),
        "df_hourly": df_results,
    }


def _check_dispatch(dispatch, available_solar, power_rating, charging_power,
                    air_capacity, tes_capacity, initial_air, initial_tes,
                    dt_hours, enforce_end_of_year_soc, *,
                    allow_solar_while_expanding=ALLOW_SOLAR_WHILE_EXPANDING,
                    allow_compression_while_expanding=ALLOW_COMPRESSION_WHILE_EXPANDING,
                    turbomachinery_full_load_only=TURBOMACHINERY_FULL_LOAD_ONLY,
                    tes_power=None):
    """Check physical balances independently of the optimization formulation."""
    p_ch = dispatch["P_Charge_Positive"].to_numpy()
    p_dis = dispatch["P_Disch"].to_numpy()
    air = dispatch["Air_kg"].to_numpy()
    tes = dispatch["TES_MWhth"].to_numpy()
    solar = dispatch["Solar_Captured_MWth"].to_numpy()
    tes_in = dispatch["TES_Charge_MWth"].to_numpy()
    tes_out = dispatch["TES_Discharge_MWth"].to_numpy()
    tol = 1e-4  # Absolute tolerance for MW/MWh results.
    air_tol = 0.1  # kg; the air bus uses a different numerical scale.
    if not np.isfinite(dispatch.to_numpy()).all():
        raise SolverOptimizationError("Dispatch contains missing or non-finite results.")
    if (np.any(p_ch < -tol) or np.any(p_dis < -tol) or np.any(solar < -tol) or
        np.any(tes_in < -tol) or np.any(tes_out < -tol) or
        np.any(p_ch > charging_power + tol) or np.any(p_dis > power_rating + tol) or
        np.any((tes_in > tol) & (tes_out > tol)) or
        np.any(air < -air_tol) or np.any(air > air_capacity + air_tol) or
        np.any(tes < -tol) or np.any(tes > tes_capacity + tol) or
        np.any(solar > available_solar.to_numpy() + tol)):
        raise SolverOptimizationError("Optimized dispatch violates a plant capacity or mode constraint.")
    if tes_power is not None and (np.any(tes_in > tes_power + tol) or np.any(tes_out > tes_power + tol)):
        raise SolverOptimizationError("TES charging/discharging exceeded its power rating.")
    if not allow_compression_while_expanding and np.any((p_ch > tol) & (p_dis > tol)):
        raise SolverOptimizationError("Compression and expansion occurred simultaneously.")
    if not allow_solar_while_expanding and np.any((p_dis > tol) & (solar > tol)):
        raise SolverOptimizationError("Solar collection and expansion occurred simultaneously.")
    if turbomachinery_full_load_only and (
        np.any((p_ch > tol) & (np.abs(p_ch - charging_power) > tol)) or
        np.any((p_dis > tol) & (np.abs(p_dis - power_rating) > tol))
    ):
        raise SolverOptimizationError("Turbomachinery operated at part load in full-load-only mode.")
    air_previous = np.r_[initial_air, air[:-1]]
    tes_previous = np.r_[initial_tes, tes[:-1]]
    air_expected = air_previous + dt_hours * (MW_TO_KGH * p_ch - p_dis / KGH_TO_MW)
    tes_expected = (tes_previous * (1 - TES_STANDING_LOSS_PER_HOUR) ** dt_hours +
                    dt_hours * (TES_CHARGING_EFFICIENCY * tes_in - tes_out))
    if not np.allclose(air, air_expected, rtol=0, atol=air_tol):
        raise SolverOptimizationError("The hourly compressed-air mass balance is inconsistent.")
    if not np.allclose(tes, tes_expected, rtol=0, atol=tol):
        raise SolverOptimizationError("The hourly TES energy balance is inconsistent.")
    if not np.allclose(solar + tes_out - tes_in, p_dis / CYCLE_EFFICIENCY, rtol=0, atol=tol):
        raise SolverOptimizationError("The thermal bus energy balance is inconsistent.")
    if enforce_end_of_year_soc and (abs(air[-1] - initial_air) > air_tol or
                                   abs(tes[-1] - initial_tes) > tol):
        raise SolverOptimizationError("The terminal storage inventory targets were not satisfied.")


# Backward compatibility: existing Streamlit imports and keyword calls still work.
# New integrations should use the English public function, simulate_plant.
def simular_planta(power_rating, TES_hours, reservoir_volume, aperture_area, pais, charging_power, **kwargs):
    """Legacy alias for simulate_plant; retained only for existing integrations."""
    return simulate_plant(power_rating, TES_hours, reservoir_volume, aperture_area, pais, charging_power, **kwargs)


# Preserve the names of the original helpers for any existing external callers.
# Positional arguments are unchanged. Original keyword names are supported here.
def calculate_IPH(dischargingPower, apertureArea, chargingPower):
    """Legacy alias for calculate_iph."""
    return calculate_iph(dischargingPower, apertureArea, chargingPower)


def calculate_capex_largeScale(dischargingPower, chargingPower, tesHours, reservoirVolume, apertureArea, nIPH, pOthers):
    """Legacy alias for calculate_capex_large_scale."""
    return calculate_capex_large_scale(dischargingPower, chargingPower, tesHours, reservoirVolume, apertureArea, nIPH, pOthers)
