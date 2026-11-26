# ASTERIx-CAESar CSP-CAES Techno-Economic Simulator

**Version 1.0.0**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

The ASTERIx-CAESar simulator is an open-source Python tool for the preliminary techno-economic assessment of hybrid concentrated solar power (CSP) and compressed air energy storage (CAES) plants. The concept combines solar thermal energy, electricity storage and industrial process heat (IPH) recovery.

Developed by Fundación CENER within the ASTERIx-CAESar Horizon Europe project, the tool enables users to explore the influence of plant sizing, geographical location and electricity market conditions on annual plant operation and financial performance.

The public Streamlit interface is intended for general users. The Python model provides access to the complete hourly simulation results, economic calculations and modelling assumptions. For further details, see [Model methodology](docs/MODEL.md).

## Features

- Interactive plant sizing for ten geographical cases.
- Hourly simulation of plant operation using PyPSA and the HiGHS optimization solver.
- Assessment of electricity trading and industrial process heat revenues.
- Estimation of capital expenditure, annual operating margin, net present value, internal rate of return, discounted payback period, capacity factor and annual electricity production.
- Financial sensitivity analysis at fixed plant dispatch, monthly storage state-of-charge charts and comparison of saved design scenarios.
- Standalone Python interface for detailed results and further research.

The tool is intended for preliminary assessment. Its results are indicative and do not replace detailed engineering design, site-specific feasibility studies or investment appraisal.

## Installation and usage

Python 3.11 is the reference version for the supplied dependencies. Clone the repository and create a virtual environment in the project directory.

```bash
git clone https://github.com/asterixcaesar/asterix-caesar-simulator.git
cd asterix-caesar-simulator
```

**Windows (Command Prompt)**

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py
```

**Linux/macOS**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py
```

Open the local URL displayed by Streamlit. Select a geographical case, define the plant design parameters and click **Run Simulation**. Once the calculation is complete, explore the results, adjust the financial assumptions or save the design for comparison with other configurations.

A full-year simulation may take several minutes, depending on the selected design and available computational resources. The sensitivity analysis does not repeat the dispatch optimization.

## Plant design inputs

The user specifies the rated electrical output, compressor electrical power, TES duration at rated output, air reservoir volume, solar-field aperture area and geographical case. Not all combinations are valid: the application checks the adopted industrial process heat correlation limit and the feasibility of the operational optimization. The available input ranges and engineering assumptions are described in [docs/MODEL.md](docs/MODEL.md).

## Input data and sources

The `data/` directory contains the electricity price and solar resource files used by the model. The electricity price scenarios refer to 2025 and are expressed in USD/MWh, as indicated in the supplied data headers.

| Geographical case | Electricity data source | Price representation |
| --- | --- | --- |
| Australia, California, Chile, Greece (mainland), Iran and Texas | [International Energy Agency (IEA)](https://www.iea.org/) | Hourly electricity price series |
| Spain | [ESIOS (Red Eléctrica)](https://www.esios.ree.es/) | Hourly electricity price series |
| Greece (Rhodes Island) | Private/internal source | Fixed price of 96.97 USD/MWh |
| Morocco | ONEE regulated tariff | Hourly tariff profile |
| South Africa | Eskom Megaflex tariff | Hourly tariff profile |

Solar resource data are based on Typical Meteorological Year (TMY) files identified in the supplied headers as [NREL NSRDB](https://nsrdb.nrel.gov/) data. The model uses direct normal irradiance (DNI), expressed in W/m². Each electricity price profile and corresponding solar resource profile contains 8,760 hourly records on a common non-leap-year reference calendar. TMY conditions represent a typical solar year, not necessarily the actual weather in 2025.

The files use elapsed simulation time in seconds in the first column and the price or DNI in the second column. The loader skips two header lines for electricity data and five for solar data. It checks the record count, finite input values and consecutive hourly timestamps. Negative electricity prices are retained. The input files must already be expressed in the required units, currency and common time convention; the model does not perform exchange-rate or time-zone conversions. Tariff-based and fixed-price cases are screening scenarios rather than necessarily observed wholesale trading prices.

Third-party data remain subject to the applicable source-specific usage and redistribution conditions. The MIT licence for the original software does not supersede these conditions.

## Python interface

The same simulation engine can be used independently of Streamlit:

```python
from asterix_caesar_model import simulate_plant

results = simulate_plant(
    power_rating=100,
    TES_hours=7,
    reservoir_volume=180_000,
    aperture_area=450_000,
    country="Chile",
    charging_power=65,
)

print(results["capex_musd"])
print(results["npv_musd"])
print(results["irr_pct"])
print(results["cf_pct"])

hourly = results["df_hourly"]
```

For a standalone run with detailed results saved locally, execute the following command from the repository root. The output directory must not already exist.

```bash
python specific_case.py --location Chile --power-rating 100 --charging-power 65 --tes-hours 7 --reservoir-volume 180000 --aperture-area 450000 --output-dir results/chile-reference
```

This creates `summary.json` and `timeseries.csv`, including the design, financial assumptions, model version and dataset fingerprint in the summary. The public website deliberately does not expose result-download functionality. Advanced operational options and time limits are documented in the `simulate_plant()` docstring.

## Repository structure

| File or directory | Description |
| --- | --- |
| `app.py` | Public Streamlit application |
| `asterix_caesar_model.py` | Simulation, operational optimization, CAPEX and financial assessment |
| `ui_helpers.py` | Shared formatting, file fingerprint and URL helpers |
| `specific_case.py` | Standalone command-line simulation |
| `data/` | Electricity price and solar resource files |
| `assets/` | Project logos, EU funding graphic and loading animation |
| `.streamlit/` | Streamlit visual configuration |
| `docs/MODEL.md` | Scientific methodology, assumptions and limitations |
| `requirements.txt` | Python dependencies |
| `CITATION.cff` | Software citation metadata |
| `CHANGELOG.md` | Public release history |
| `LICENSE` | MIT licence |

## Scientific references and citation

The scientific background and empirical correlations are based on the research conducted within ASTERIx-CAESar. The relevant publications are listed in [docs/MODEL.md](docs/MODEL.md#references).

If you use this software in your research, please cite the corresponding software release using [CITATION.cff](CITATION.cff) and acknowledge the relevant scientific publications. Cite the version or Git tag used to reproduce your results.

**Software authors:** Javier Baigorri and Fritz Zaversky (Fundación CENER, Spain).

## Licence

Copyright (c) 2026 Fundación CENER.

The original software and associated documentation are released under the MIT License. See [LICENSE](LICENSE). Third-party software, datasets and logos are subject to their respective licences and usage conditions.

## Funding and acknowledgement

![Funded by the European Union](assets/EN_FundedbytheEU_RGB_POS.png)

This software has been developed within the ASTERIx-CAESar project, funded by the European Union under the Horizon Europe research and innovation programme (Grant Agreement No. 101122231).

Project website: <https://asterix-caesar.eu/>  
Project information: <https://cordis.europa.eu/project/id/101122231>

Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or the granting authority. Neither the European Union nor the granting authority can be held responsible for them.
