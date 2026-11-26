# Model methodology and limitations

This document describes the simplified CSP-CAES simulation model implemented in the ASTERIx-CAESar techno-economic assessment tool (version 1.0.0). The model evaluates a fixed plant design over a representative year, optimizes hourly operation and estimates financial performance. It is intended for preliminary screening rather than detailed component design or investment appraisal.

The Streamlit application and the standalone Python interface use the same simulation engine, [`asterix_caesar_model.py`](../asterix_caesar_model.py). The source code contains the exact coefficients, cost correlations and mathematical constraints. Input data sources and installation instructions are described in the [README](../README.md). The scientific background is provided in the [references](#references) below.

## 1. System concept and design inputs

The plant integrates an electrically driven compression train, a compressed-air reservoir, an expansion train with a bottoming cycle, a solar field and receiver, thermal energy storage (TES) and industrial process heat (IPH) recovery. Solar thermal energy supplies the heat required for expansion, either directly or from the TES. Electricity is purchased during compression and sold during expansion. Recoverable process heat is valued as an additional revenue stream.

| Design parameter | Unit | Available website range |
| --- | --- | ---: |
| Rated electrical output of the expansion train | MWe | 1-400 |
| Rated electrical input of the compressor | MWe | 0.5-65 |
| TES duration at rated electrical output | h | 2-24 |
| Air reservoir volume | m³ | 100-600,000 |
| Solar-field aperture area | m² | 1,000-800,000 |

These are interface ranges, not independently established engineering validity ranges. The nominal recoverable IPH capacity must be below 40 MWth by default. A configuration within the available input ranges may still violate this limit or be operationally infeasible.

Two plant-size configurations are considered, with a transition at 20 MWe. The small-scale configuration (20 MWe and below) uses artificial above-ground compressed-air storage and an Organic Rankine Cycle (ORC) as the bottoming cycle, with the corresponding cost correlations. The large-scale configuration (above 20 MWe) uses the large-scale storage and bottoming-cycle correlations. The selected reservoir technology, maximum pressure and component-cost correlations must be consistent across the model.

## 2. Physical model

### 2.1. Main assumptions

| Parameter | Adopted value |
| --- | --- |
| Reference simulation calendar | 2025; 8,760 hourly snapshots |
| Air reservoir pressure, small scale (<= 20 MWe) | 40-150 bar absolute |
| Air reservoir pressure, large scale (>20 MWe) | 40-80 bar absolute |
| Air reservoir temperature | 303.15 K (isothermal) |
| Specific gas constant of air | 287.05 J/(kg K) |
| Compression conversion factor | 6,600 kg/h per MW of electrical input |
| Expansion conversion factor | 0.0001975 MW per kg/h of expanded air |
| Effective thermal-to-electric conversion efficiency | 50% |
| TES charging efficiency | 97% |
| TES discharging efficiency | 100% |
| TES standing loss | 0.1% per hour |
| Initial air reservoir and TES state of charge | 50% |
| Final air reservoir and TES state of charge | Equal to initial values by default |

The conversion factors are fixed screening-model parameters; they do not constitute a detailed representation of the compression and expansion trains at different reservoir pressures and part-load conditions.

### 2.2. Compressed-air storage

The reservoir is represented by an isothermal ideal-gas model. The usable air mass between the minimum and maximum operating pressures is:

$$
m_{\mathrm{usable}} = \frac{(p_{\max}-p_{\min})\,10^5\,V}{R T}
$$

where pressures are in bar absolute, reservoir volume $V$ is in m³, temperature $T$ is in K, and the result is in kg. The air mass associated with the minimum pressure is excluded from the usable storage capacity.

For compressor electrical power $P_{\mathrm{ch}}$ and expander electrical output $P_{\mathrm{dis}}$, both expressed as non-negative values in MW, the discrete air mass balance is:

$$
m_t=m_{t-1}+\Delta t\left(6600\,P_{\mathrm{ch},t}-\frac{P_{\mathrm{dis},t}}{1.975\times10^{-4}}\right)
$$

Here $\Delta t$ is the time step in hours. The air inventory is bounded between zero and the usable mass capacity. Air leakage is not modelled separately.

### 2.3. Solar thermal energy

The combined solar-field and receiver efficiency is estimated using an empirical aperture-area correlation:

$$
\eta_{\mathrm{solar}}=-0.02026\ln(A)+0.79844
$$

The numerical value of aperture area $A$ must be expressed in m², consistent with the correlation. Available solar thermal power is:

$$
\dot Q_{\mathrm{solar,avail},t}=\frac{\mathrm{DNI}_t\,A\,\eta_{\mathrm{solar}}}{10^6}
$$

DNI is expressed in W/m² and the calculated power in MWth. Solar heat actually collected is an optimization variable bounded by the available thermal power. Surplus solar energy may be curtailed.

### 2.4. Thermal energy storage and expansion

TES energy capacity is defined by the rated expansion output $P_{\mathrm{rated}}$, nominal TES duration $h_{\mathrm{TES}}$ and the effective conversion efficiency $\eta_{\mathrm{cycle}}=0.50$:

$$
E_{\mathrm{TES,max}}=\frac{h_{\mathrm{TES}}P_{\mathrm{rated}}}{\eta_{\mathrm{cycle}}}
$$

The result is in MWhth. TES charge and discharge power are limited by a common nominal thermal power, defined as the greater of the rated expansion thermal demand and the maximum available solar thermal power in the selected year.

The TES energy balance is:

$$
E_t=(1-0.001)^{\Delta t}E_{t-1}+\Delta t\left(0.97\,\dot Q_{\mathrm{store},t}-\dot Q_{\mathrm{dispatch},t}\right)
$$

The thermal bus balance is:

$$
\dot Q_{\mathrm{solar},t}+\dot Q_{\mathrm{dispatch},t}-\dot Q_{\mathrm{store},t}=\frac{P_{\mathrm{dis},t}}{\eta_{\mathrm{cycle}}}
$$

The backend accounts for the standing loss at the beginning of each rolling window through its initial-inventory correction. This treatment must remain consistent with the PyPSA storage balance implementation when upgrading dependencies.

### 2.5. Industrial process heat recovery

Recoverable IPH is estimated from empirical correlations for the charging and discharging modes. The nominal compression heat recovery is:

$$
\dot Q_{\mathrm{IPH,ch,nom}}=0.5857\,P_{\mathrm{ch,nom}}+0.3284
$$

Nominal discharging heat recovery consists of an aperture-dependent atmospheric heat recovery term and a power-dependent term. The atmospheric contribution is $0.0000465 A-0.0151629$ MWth. The power-dependent correlation differs between small- and large-scale plants. See `calculate_iph()` for the exact expressions.

Hourly recoverable IPH scales linearly with the operating power fraction in each mode:

$$
\dot Q_{\mathrm{IPH},t}=\dot Q_{\mathrm{IPH,ch,nom}}\frac{P_{\mathrm{ch},t}}{P_{\mathrm{ch,nom}}}+\dot Q_{\mathrm{IPH,dis,nom}}\frac{P_{\mathrm{dis},t}}{P_{\mathrm{rated}}}
$$

With non-simultaneous compression and expansion, the nominal IPH capacity is the greater of the charging and discharging heat capacities. The default validity limit is strictly below 40 MWth. The charging-side limit alone corresponds to a compressor rating below approximately 67.73 MWe; the discharging-side limit also depends on aperture area and rated power.

The base scenario assumes that all recoverable heat is sold at 50 USD/MWhth. Heat sales are included in the optimization objective by default. A separate demand-constrained industrial heat network, heat quality variation and thermal offtake schedule are not modelled. Heat revenue can be disabled through the Python API without removing the installed IPH equipment from CAPEX.

## 3. Operational optimization

The energy system is represented in PyPSA using electrical, compressed-air and thermal buses. The air reservoir is a `Store`, TES is a `StorageUnit`, and the compressor and expansion train are `Link` components. A bidirectional market interface represents electricity purchases and sales. The solar resource is represented by a dispatchable thermal `Generator` limited by available solar heat.

HiGHS solves the resulting mixed-integer linear optimization problems. Binary operating-mode constraints prevent incompatible modes of operation.

| Operational assumption | Default setting |
| --- | --- |
| Simultaneous compression and expansion | Not permitted |
| Solar collection during expansion | Permitted |
| Turbomachinery part-load operation | Permitted, with fixed conversion factors |
| Simultaneous TES charging and discharging | Not permitted |
| IPH sales included in dispatch objective | Yes |
| End-of-year inventory conservation | Enforced for both stores |

### 3.1. Rolling horizon

Each annual simulation is divided into optimization windows of 72 h, with the first 48 h committed to the annual operating schedule. The committed terminal air and TES inventories become the initial conditions of the following window. The final short remainder is merged into the last optimization window, resulting in 182 solver calls for the default 8,760-hour year.

The initial and final air and TES inventories are equal by default. The terminal constraints are applied to the final committed window. This avoids systematic depletion of initial stored energy in the annual financial assessment; however, the final target may be infeasible for some plant configurations or input series.

### 3.2. Objective and solver settings

The default objective maximizes electricity trading margin and industrial process heat revenue over each optimization window. Equivalently, the solver minimizes electricity purchase costs minus electricity sales and heat revenues, with a small positive solar collection cost of $10^{-6}$ USD/MWhth to discourage unnecessary heat collection. This numerical tie-break cost is excluded from the reported financial results.

| Solver setting | Default |
| --- | --- |
| Solver | HiGHS |
| Solver interface | Direct |
| Solver threads | 1 |
| Relative MIP gap | 1% per optimization window |
| Website solver time limit | 60 s per window |
| Website overall simulation budget | 600 s, cooperative time limit |

The time limits are application resource settings, not physical assumptions. The Python interface allows time limits to be specified independently. Incomplete or non-optimal solver terminations are not accepted as successful annual results. The relative MIP gap applies to each window, not to the annual NPV. The rolling horizon assumes perfect knowledge of prices and solar availability within each optimization window and does not guarantee a globally optimal annual dispatch.

## 4. Capital expenditure and financial assessment

### 4.1. Capital expenditure

The CAPEX model is based on empirical component-cost correlations. The direct investment includes the high- and low-pressure turbines, generator and gearbox, bottoming cycle, compressor train, air reservoir, heat exchangers, TES, solar field, receiver, tower, blower and IPH recovery system.

The storage and bottoming-cycle cost correlations depend on the plant-size configuration. The exact cost correlations, including the small-scale ORC calculation, are implemented in `calculate_capex_breakdown()`. The cost escalation index (CEPCI) used by the correlations is 796.3. See the referenced publications and source code for the adopted cost basis and component-specific multipliers.

The gross plant CAPEX includes indirect costs, assumed to represent 30% of the total investment:

$$
C_{\mathrm{gross}}=\frac{C_{\mathrm{direct}}}{1-0.30}
$$

The correlation-based CAPEX is an indicative screening estimate rather than a supplier quotation or a site-specific investment budget.

### 4.2. Financial assumptions

| Parameter | Base value |
| --- | --- |
| Currency | USD |
| IPH selling price | 50 USD/MWhth |
| Indirect CAPEX | 30% of gross CAPEX |
| Fixed annual O&M | 1.5% of direct CAPEX |
| Project lifetime | 30 years |
| Real discount rate | 6% |
| Investment subsidy | 0% |

Annual electricity sales and purchasing costs are calculated by integrating the optimized hourly electrical powers multiplied by the corresponding price. The base model uses the same input electricity price for imports and exports. The operating margin is:

$$
M_{\mathrm{op}}=R_{\mathrm{electricity}}+R_{\mathrm{IPH}}-C_{\mathrm{electricity,purchase}}
$$

This margin excludes fixed O&M. Annual net cash flow and net initial investment are:

$$
CF_{\mathrm{annual}}=M_{\mathrm{op}}-0.015\,C_{\mathrm{direct}}
$$

$$
I_0=C_{\mathrm{gross}}(1-s)
$$

where $s$ is the subsidy fraction. Subsidies reduce the initial investment but do not reduce fixed O&M expenditure. Assuming a constant real annual cash flow, net present value is:

$$
NPV=-I_0+\sum_{y=1}^{N}\frac{CF_{\mathrm{annual}}}{(1+r)^y}
$$

where $N$ is the project lifetime and $r$ the real discount rate. The internal rate of return (IRR) is the rate for which NPV equals zero. The discounted payback period (DPP) is obtained from cumulative discounted cash flow, with linear interpolation within the year of investment recovery.

The financial calculation excludes taxes, debt financing, degradation, equipment replacements and residual value. Undefined IRR values are returned as `None` and displayed as `N/A`; a discounted payback not achieved within the project lifetime is displayed as `Not reached`.

Electricity-only diagnostic financial indicators in the Python results exclude IPH revenue while retaining the same dispatch and installed plant CAPEX. They are not the results of a separate design or dispatch optimization without heat recovery.

### 4.3. Financial sensitivity analysis

The website allows the user to vary electricity market prices, subsidy, discount rate and the annual O&M rate. Both the electricity purchase cost and sales revenue are scaled by the same electricity price multiplier; IPH revenue remains unchanged. The O&M rate applies to direct CAPEX before subsidy.

The sensitivity analysis recalculates financial indicators using the previously optimized hourly plant operation. It does not re-optimize operation for the modified electricity prices or financial assumptions. For a new operating strategy, the input data or model settings must be changed and the simulation rerun.

## 5. Performance indicators and validation

Annual electrical generation and consumption are obtained by integrating the corresponding hourly powers. The capacity factor is annual electricity production divided by the maximum possible annual generation at rated electrical output. Monthly state-of-charge charts display the stored thermal energy and usable compressed-air mass as percentages of their respective capacities.

The electricity production-to-consumption ratio is a diagnostic indicator, not thermodynamic round-trip efficiency. Under the fixed air conversion factors and annual air inventory conservation, it is approximately 130.35% for nonzero cycling. This is possible because external solar thermal energy is supplied to the system. The public website therefore displays annual electricity production rather than this essentially fixed ratio.

The backend checks the annual dispatch independently of the optimization formulation. These checks cover the following:

- Complete and finite hourly results and valid component power limits.
- Air mass and TES energy balances, including standing losses.
- Thermal bus energy balance and limits on available solar energy.
- Storage inventory bounds and mutually exclusive operating modes.
- Annual initial and final inventory equality, when enforced.

A successful simulation satisfies the implemented numerical checks; it does not by itself establish the engineering feasibility of a real plant or validate the empirical correlations for every possible design. The standalone runner in `specific_case.py` exports the detailed hourly results and a JSON summary for local analysis and reproduction.

## 6. Main limitations

- Isothermal ideal-gas storage and constant conversion factors simplify the physical behaviour of the plant. Pressure-dependent performance, dynamic thermal response, auxiliary consumption, start-up losses, minimum stable loads, ramp rates and equipment outages are not modelled separately.
- The plant-size transition at 20 MWe reflects a change in the assumed storage and bottoming-cycle technologies, not a continuous engineering cost relationship. Small-scale and large-scale cost and performance estimates require the corresponding engineering context.
- The IPH correlations and assumed full heat offtake do not represent a specific contracted heat market or industrial demand profile.
- The representative year combines electricity price or tariff scenarios with typical-year solar resource data. It does not reproduce the actual joint weather and price conditions of a historical year. Prices and solar data must already be aligned to a common hourly convention.
- The model does not explicitly account for site-specific geology, land constraints, permitting, grid connection conditions, transaction charges or detailed plant financing. Results depend on the adopted empirical cost correlations, electricity price series and financial assumptions.
- Hourly rolling-horizon optimization and fixed-dispatch financial sensitivity are suitable for comparative screening, but do not establish a guaranteed operating strategy or investment return.

## References

1. J. Baigorri, A. Federici, T. Kubikova, T. Du Toit, P. Rodríguez-deArriba, C. Salvini, D. Sánchez and F. Zaversky, "Modeling of an innovative integration of compressed air energy storage (CAES) with high-temperature concentrated solar power (CSP): A comprehensive use-case study," *Journal of Energy Storage*, 132 (2025), 117678. <https://doi.org/10.1016/j.est.2025.117678>.
2. J. Baigorri, P. Rodríguez-deArriba, T. Kubikova, T. Du Toit, J. Romani, A. L. Ávila-Marín, D. Sánchez and F. Zaversky, "Evaluating a compressed air energy storage (CAES) system with high-temperature concentrated solar power (CSP): Techno-economic and sensitivity analysis," *Journal of Energy Storage*, 180, Part A (2026), 124072. <https://doi.org/10.1016/j.est.2026.124072>.

Optimization software: [PyPSA](https://pypsa.org/) and [HiGHS](https://highs.dev/). For software citation metadata, see [`CITATION.cff`](../CITATION.cff).
