"""Command-line entry point for AIDEOM-VN.

Run:
    python run_all.py

Optional:
    python run_all.py --with-rl
"""
from __future__ import annotations

import argparse

from src.scenario_pipeline import export_outputs, run_all_modules
from src.visualization import save_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AIDEOM-VN models")
    parser.add_argument("--with-rl", action="store_true", help="also train the Q-learning module")
    args = parser.parse_args()
    outputs = run_all_modules(include_rl=args.with_rl)
    export_outputs(outputs)
    figures = save_all_figures(outputs)
    print("AIDEOM-VN run completed")
    print("GDP 2030 scenarios:")
    print(outputs["scenarios"][["scenario", "GDP_2030", "D_2030", "AI_2030", "H_2030"]])
    print("Saved figures:")
    for figure in figures:
        print(" -", figure)


if __name__ == "__main__":
    main()
