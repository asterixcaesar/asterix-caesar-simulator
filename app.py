# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fundación CENER
"""Public ASTERIx-CAESar simulator: a lightweight interface to the research model."""

import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import asterix_caesar_model as model
from ui_helpers import file_fingerprint, format_value, payback_display, valid_repository_url

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGGER = logging.getLogger(__name__)

# Set this environment variable at deployment, or insert the approved URL here.
GITHUB_REPOSITORY_URL = os.getenv("ASTERIX_GITHUB_URL", "https://github.com/asterixcaesar/asterix-caesar-simulator").strip()
# Provisional resource limits; benchmark against the target server before release.
WINDOW_TIMEOUT_SECONDS = 60.0
ANNUAL_TIMEOUT_SECONDS = 600.0
CACHE_MAX_ENTRIES = 5  # Per user session, not a server-wide memory limit.
CACHE_TTL_SECONDS = 86400

LOCATION_OPTIONS = {
    "Australia": "Australia",
    "California": "California",
    "Chile": "Chile",
    "Greece — Mainland Grid": "Greece_MainlandGrid",
    "Greece — Rhodes Island": "Greece_RhodesIsland",
    "Iran": "Iran",
    "Morocco": "Morocco",
    "South Africa": "Southafrica",
    "Spain": "Spain",
    "Texas": "Texas",
}
SENSITIVITY_DEFAULTS = {
    "sens_price": 100,
    "sens_subsidy": int(model.SUBSIDY_PERCENT),
    "sens_disc": round(100 * model.REAL_DISCOUNT_RATE),
    "sens_om": 100 * model.O_AND_M_FRACTION_OF_DIRECT_CAPEX,
}


def reset_sensitivity() -> None:
    """Reset the financial what-if controls to the model's base assumptions."""
    for key, value in SENSITIVITY_DEFAULTS.items():
        st.session_state[key] = value


def dataset_fingerprint(location: str) -> str:
    """Check the actual dataset before making an expensive solver call."""
    price_path, solar_path = model._resolve_input_files(location, None)
    hours = len(model._reference_snapshots())
    model._load_input_column(price_path, 2, hours, "Electricity prices")
    model._load_input_column(solar_path, 5, hours, "Solar resource")
    return file_fingerprint(price_path, solar_path)


def render_progress(slot, animation: str, completed: int, total: int) -> None:
    """Display measured progress within the existing loading animation."""
    percentage = min(100, int(100 * completed / total))
    slot.markdown(animation.replace("__PROGRESS_VALUE__", str(percentage)), unsafe_allow_html=True)


def simulation_signature(params: dict, dataset_hash: str) -> tuple:
    """Identify the exact design, backend source and input dataset."""
    return (
        model.MODEL_VERSION, file_fingerprint(Path(model.__file__)), dataset_hash,
        tuple(sorted(params.items())), WINDOW_TIMEOUT_SECONDS, ANNUAL_TIMEOUT_SECONDS,
    )


def run_model(params: dict, key: tuple) -> dict:
    """Reuse successful session results; show progress only on a cache miss."""
    cache = st.session_state.simulation_cache
    now = time.monotonic()
    for cached_key, (created, _) in list(cache.items()):
        if now - created > CACHE_TTL_SECONDS:
            del cache[cached_key]
    if key in cache:
        cache.move_to_end(key)
        return cache[key][1]

    slot = st.empty()
    try:
        animation = (ASSETS_DIR / "loading_animation.html").read_text(encoding="utf-8")
        last_percentage = -5

        def progress(completed: int, total: int) -> None:
            nonlocal last_percentage
            percentage = int(100 * completed / total)
            # Throttle browser updates to avoid restarting the SVG each hour.
            if percentage >= last_percentage + 5 or completed == total:
                render_progress(slot, animation, completed, total)
                last_percentage = percentage

        progress(0, 1)
        result = model.simulate_plant(
            power_rating=params["power_rating"],
            TES_hours=params["tes_hours"],
            reservoir_volume=params["reservoir_volume"],
            aperture_area=params["aperture_area"],
            country=params["location"],
            charging_power=params["charging_power"],
            solver_time_limit_seconds=WINDOW_TIMEOUT_SECONDS,
            overall_time_limit_seconds=ANNUAL_TIMEOUT_SECONDS,
            progress_callback=progress,
        )
        if result["status"] != ("ok", "optimal"):
            raise model.SolverOptimizationError("The simulation did not complete successfully.")
        cache[key] = (time.monotonic(), result)
        while len(cache) > CACHE_MAX_ENTRIES:
            cache.popitem(last=False)
        return result
    finally:
        slot.empty()


def main() -> None:
    st.set_page_config(page_title="CSP-CAES Simulator", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    h1 {color: #F49000;}
    h2, h3 {color: #6B91DD;}
    [data-testid="stMetricValue"] {color: #6B91DD;}
    </style>
    """, unsafe_allow_html=True)

    for name in ("logo.png", "logo.jpg"):
        logo = ASSETS_DIR / name
        if logo.is_file():
            st.sidebar.image(str(logo), width="stretch")
            break

    st.title("Techno-Economic Analysis of ASTERIx-CAESar Plant")
    st.markdown("Adjust the design parameters of a Concentrated Solar Power plant integrated with Compressed Air Energy Storage.")

    for key, default in {
        "results": None, "sim_params": None, "sim_signature": None, "scenarios": [],
        "simulation_cache": OrderedDict(), **SENSITIVITY_DEFAULTS,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.sidebar.header("Design Parameters")
    location_display = st.sidebar.selectbox("Select Location", list(LOCATION_OPTIONS))
    location = LOCATION_OPTIONS[location_display]
    power_rating = st.sidebar.number_input("Electrical Power Rating (MW)", min_value=1, max_value=400, value=100, step=10)
    tes_hours = st.sidebar.slider("Thermal Energy Storage (Hours)", min_value=2, max_value=24, value=8)
    reservoir_volume = st.sidebar.number_input("Air Reservoir Volume (m³)", min_value=100, max_value=600000, value=200000, step=1000)
    aperture_area = st.sidebar.number_input("Solar Aperture Area (m²)", min_value=1000, max_value=800000, value=450000, step=5000)
    charging_power = st.sidebar.number_input("Compressor Charging Power (MW)", min_value=0.5, max_value=65.0, value=65.0, step=5.0, format="%.1f")
    params = dict(location=location, location_display=location_display, power_rating=power_rating,
                  tes_hours=tes_hours, reservoir_volume=reservoir_volume,
                  aperture_area=aperture_area, charging_power=charging_power)

    # Retain supplied ranges; apply only the established model validity limit.
    _, _, nominal_iph = model.calculate_iph(power_rating, aperture_area, charging_power)
    valid_design = model.ALLOW_HIGHER_IPH or nominal_iph < model.IPH_MAX_CORRELATION_POWER_MW
    if not valid_design:
        st.sidebar.error(f"Recoverable heat is {nominal_iph:.1f} MWth. Choose a design below the {model.IPH_MAX_CORRELATION_POWER_MW:g} MWth model limit.")

    dataset_hash = None
    try:
        dataset_hash = dataset_fingerprint(location)
    except (model.InputDataError, OSError) as exc:
        LOGGER.warning("Dataset preflight failed for %s: %s", location, exc)
        st.sidebar.warning(f"The electricity or solar dataset for {location_display} is incomplete or unavailable. Please select another location.")

    signature = simulation_signature(params, dataset_hash) if dataset_hash is not None else None
    if st.sidebar.button("Run Simulation", disabled=not valid_design or signature is None):
        st.session_state.results = None
        st.session_state.sim_params = None
        st.session_state.sim_signature = None
        try:
            result = run_model(params, signature)
        except model.InvalidDesignError as exc:
            st.error(str(exc))
        except model.InputDataError:
            LOGGER.exception("Invalid simulation dataset")
            st.error("The selected input dataset is incomplete or invalid.")
        except model.SolverOptimizationError:
            LOGGER.exception("Optimization failed")
            st.error("The simulation could not find a valid solution within the configured limits. Try a different plant design.")
        except Exception:
            LOGGER.exception("Unexpected simulation failure")
            st.error("The simulation encountered a technical problem. Please try again later.")
        else:
            st.session_state.results = result
            st.session_state.sim_params = params.copy()
            st.session_state.sim_signature = signature
            reset_sensitivity()

    st.sidebar.divider()
    st.sidebar.caption(f"ASTERIx-CAESar | Version {model.MODEL_VERSION}")
    st.sidebar.caption("Developed within the ASTERIx-CAESar project. No. 101122231. Funded by the European Union under the Horizon Europe programme.")

    for name in ("EN_FundedbytheEU_RGB_POS.png", "EN_FundedbytheEU_RGB_POS.jpg"):
        fundedLogo = ASSETS_DIR / name
        if fundedLogo.is_file():
            st.sidebar.image(str(fundedLogo), width="stretch")
            break

    if valid_repository_url(GITHUB_REPOSITORY_URL):
        st.sidebar.markdown(f"[Open source code and documentation on GitHub]({GITHUB_REPOSITORY_URL})")
    else:
        st.sidebar.caption("GitHub repository link pending publication.")
    st.sidebar.caption("Indicative results for preliminary assessment.")

    results = st.session_state.results
    if results is not None:
        simulated_params = st.session_state.sim_params
        if signature != st.session_state.sim_signature:
            st.sidebar.warning("Design parameters or input data changed. Run the simulation to update the displayed results.")
        else:
            st.success("Simulation completed successfully.")
        for warning in results.get("warnings", []):
            st.warning(warning)
        show_results(results, simulated_params)

    if st.session_state.scenarios:
        st.divider()
        st.subheader("Scenario Comparison")
        st.dataframe(pd.DataFrame(st.session_state.scenarios).drop(columns=["_label"]), width="stretch")
        if st.button("Clear all saved scenarios"):
            st.session_state.scenarios = []
            st.rerun()


def show_results(results: dict, params: dict) -> None:
    st.subheader("Key Performance Indicators")
    cols = st.columns(4)
    cols[0].metric("CAPEX", "$ " + format_value(results["capex_musd"], suffix=" M"), help="Total upfront investment (USD).")
    cols[1].metric("NPV", "$ " + format_value(results["npv_musd"], suffix=" M"), help=f"Net present value over {model.PROJECT_LIFE_YEARS} years at a {100 * model.REAL_DISCOUNT_RATE:g}% real discount rate (USD).")
    cols[2].metric("IRR", format_value(results["irr_pct"], suffix=" %"), help="Internal rate of return.")
    cols[3].metric("Payback (DPP)", payback_display(results["dpp_years"]), help="Discounted payback period. Not reached means the investment is not recovered within the project lifetime.")
    cols = st.columns(4)
    cols[0].metric("Annual Operating Margin", "$ " + format_value(results["operating_margin_usd"] / 1e6, suffix=" M"), help="Electricity and heat sales minus electricity purchases, before fixed O&M (USD/year).")
    cols[1].metric("Capacity Factor", format_value(results["cf_pct"], suffix=" %"), help="Annual electricity production divided by the maximum possible annual production at rated power.")
    cols[2].metric("Annual Electricity Production", format_value(results["energy_discharged_mwh"] / 1000, 1, " GWh"))

    label = (f"{params['location_display']} | {params['power_rating']} MW | {params['tes_hours']}h TES | "
             f"{params['aperture_area'] / 1000:g}k m² | {params['reservoir_volume'] / 1000:g}k m³ | {params['charging_power']} MW ch.")
    if st.button(f'Save scenario: "{label}"'):
        if any(row["_label"] == label for row in st.session_state.scenarios):
            st.info("This scenario is already in the comparison table.")
        else:
            st.session_state.scenarios.append({
                "_label": label, "Location": params["location_display"],
                "Power Rating (MW)": params["power_rating"], "TES (h)": params["tes_hours"],
                "Aperture (k m²)": params["aperture_area"] / 1000,
                "Reservoir (k m³)": params["reservoir_volume"] / 1000,
                "Charging Power (MW)": params["charging_power"],
                "CAPEX (M USD)": round(results["capex_musd"], 2),
                "NPV (M USD)": round(results["npv_musd"], 2),
                "IRR (%)": format_value(results["irr_pct"]),
                "DPP (years)": payback_display(results["dpp_years"]),
                "Operating Margin (M USD/year)": round(results["operating_margin_usd"] / 1e6, 2),
                "CF (%)": round(results["cf_pct"], 2),
                "Production (GWh)": round(results["energy_discharged_mwh"] / 1000, 1),
            })
            st.success("Scenario saved. Scroll down to see the comparison table.")

    st.divider()
    show_sensitivity(results)
    st.divider()
    show_soc(results)


def show_sensitivity(results: dict) -> None:
    st.subheader("Financial Sensitivity Analysis")
    st.markdown("Adjust the economic assumptions to see how financial indicators respond, keeping the previously optimized plant operation unchanged.")
    st.button("Reset to base values", on_click=reset_sensitivity)
    cols = st.columns(4)
    price_mult = cols[0].slider("Electricity Market Price (% of base)", 50, 200, step=5, key="sens_price") / 100
    subsidy_pct = cols[1].slider("Investment Grant / Subsidy (%)", 0, 60, step=5, key="sens_subsidy")
    discount_rate = cols[2].slider("Discount Rate (%)", 3, 15, step=1, key="sens_disc") / 100
    om_rate = cols[3].slider("Annual O&M Cost (% of Direct CAPEX)", 0.5, 3.0, step=0.5, key="sens_om", help="Annual fixed O&M as a share of direct investment before subsidy.") / 100
    for col, key in zip(cols, SENSITIVITY_DEFAULTS):
        col.caption(f"Base: {SENSITIVITY_DEFAULTS[key]:g} %")

    margin = (results["electricity_revenue_usd"] - results["charging_cost_usd"]) * price_mult + results["iph_revenue_usd"]
    npv, irr, dpp = model.calculate_financial_metrics(
        margin, results["capex_musd"] * 1e6,
        operation_maintenance_factor=om_rate, real_discount_rate=discount_rate,
        subsidy_percent=subsidy_pct, direct_capex=results["direct_capex_musd"] * 1e6,
    )
    npv_delta = npv / 1e6 - results["npv_musd"]
    irr_pct = None if irr is None else 100 * irr
    irr_delta = None if irr_pct is None or results["irr_pct"] is None else irr_pct - results["irr_pct"]
    margin_delta = (margin - results["operating_margin_usd"]) / 1e6
    cols = st.columns(4)
    cols[0].metric("Adjusted NPV", "$ " + format_value(npv / 1e6, suffix=" M"), delta=f"{npv_delta:+.2f} M vs. base" if abs(npv_delta) >= 0.005 else None)
    cols[1].metric("Adjusted IRR", format_value(irr_pct, suffix=" %"), delta=f"{irr_delta:+.2f} pp vs. base" if irr_delta is not None and abs(irr_delta) >= 0.005 else None)
    cols[2].metric("Adjusted DPP", payback_display(dpp))
    cols[3].metric("Adjusted Operating Margin", "$ " + format_value(margin / 1e6, suffix=" M"), delta=f"{margin_delta:+.2f} M vs. base" if abs(margin_delta) >= 0.005 else None, help="Operating margin before fixed O&M (USD/year).")


def show_soc(results: dict) -> None:
    st.subheader("State of Charge Dynamics")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    selected_month = st.selectbox("Select Month for Chart Visualization", months)
    df = results["df_hourly"]
    df = df.loc[df.index.month == months.index(selected_month) + 1]
    fig = go.Figure()
    for column, name, color in [("SoC_Thermal_%", "Thermal TES", "#F49000"), ("SoC_Air_%", "Air Reservoir", "#6B91DD")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], name=name, line=dict(color=color, width=2), hovertemplate=f"<b>{name}:</b> %{{y:.0f}}%<extra></extra>"))
    fig.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="State of Charge (%)", yaxis=dict(range=[0, 105]),
                      legend=dict(orientation="v", x=0.01, y=0.97, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.95)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1, font=dict(size=14, color="#262730")),
                      margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#262730"))
    fig.update_xaxes(showgrid=True, gridcolor="#E2E5EA")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E5EA")
    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
