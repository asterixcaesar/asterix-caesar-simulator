# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fundación CENER
"""Optional local command-line runner using exactly the website simulation engine.

No export functionality is exposed by the public Streamlit interface.
"""

import argparse
import json
from pathlib import Path

import asterix_caesar_model as model
from ui_helpers import file_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", required=True, help="Dataset identifier, e.g. Chile or Spain")
    parser.add_argument("--power-rating", type=float, required=True, help="MWe")
    parser.add_argument("--charging-power", type=float, required=True, help="MWe")
    parser.add_argument("--tes-hours", type=float, required=True)
    parser.add_argument("--reservoir-volume", type=float, required=True, help="m3")
    parser.add_argument("--aperture-area", type=float, required=True, help="m2")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for local results")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory to avoid overwriting existing results.")
    design = {
        "power_rating": args.power_rating, "charging_power": args.charging_power,
        "TES_hours": args.tes_hours, "reservoir_volume": args.reservoir_volume,
        "aperture_area": args.aperture_area, "country": args.location,
    }
    paths = model._resolve_input_files(args.location, args.data_dir)
    fingerprint = file_fingerprint(*paths)

    def progress(completed: int, total: int) -> None:
        print(f"\rCommitted {completed}/{total} hours ({100 * completed / total:.1f}%)", end="", flush=True)

    result = model.simulate_plant(**design, data_dir=args.data_dir, progress_callback=progress)
    print()
    summary = {key: value for key, value in result.items() if key != "df_hourly"}
    summary.update({"design": design, "dataset_files": [p.name for p in paths],
                    "dataset_fingerprint": fingerprint, "reference_year": model.REFERENCE_YEAR,
                    "financial_assumptions": {"discount_rate": model.REAL_DISCOUNT_RATE,
                    "project_life_years": model.PROJECT_LIFE_YEARS,
                    "subsidy_percent": model.SUBSIDY_PERCENT,
                    "om_fraction_of_direct_capex": model.O_AND_M_FRACTION_OF_DIRECT_CAPEX,
                    "iph_heat_price_usd_mwhth": model.IPH_HEAT_PRICE}})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result["df_hourly"].to_csv(args.output_dir / "timeseries.csv", index_label="timestamp")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
