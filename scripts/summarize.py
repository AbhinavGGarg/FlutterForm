"""Consolidate eval JSONs from results_cmp/ into the headline comparison table.

    python scripts/summarize.py results_cmp
"""

import json
import sys
from pathlib import Path


def g(d, *ks, default=None):
    for k in ks:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default


def pct(x):
    return "  n/a " if x is None else f"{x:6.2f}%"


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results_cmp")

    print("\n" + "=" * 72)
    print("FlutterForm vs black-box MLP  —  held-out comparison")
    print("=" * 72)

    indist = None
    extrap = None
    if (root / "indist.json").exists():
        indist = json.loads((root / "indist.json").read_text())
    if (root / "extrap.json").exists():
        extrap = json.loads((root / "extrap.json").read_text())

    print(f"\n{'metric':<34}{'FlutterForm':>14}{'MLP baseline':>16}")
    print("-" * 72)

    def row(label, ff, bl):
        print(f"{label:<34}{ff:>14}{bl:>16}")

    if indist:
        ff, bl = indist.get("flutterform"), indist.get("baseline")
        print("IN-DISTRIBUTION (random val split)")
        row("  flutter-speed median err", pct(g(ff, "vf_median_%")), pct(g(bl, "vf_median_%")))
        row("  within 10%", pct(g(ff, "within_10%")), pct(g(bl, "within_10%")))
        row("  mode-ID accuracy", pct(g(ff, "mode_id_acc_%")),
            "  n/a (scalar)")
        row("  full V-g/V-f trajectory", "   yes", "   no (scalar)")

    if extrap:
        ff, bl = extrap.get("flutterform"), extrap.get("baseline")
        print("\nEXTRAPOLATION (train mu<40, test mu>=40 — unseen region)")
        row("  flutter-speed median err", pct(g(ff, "vf_median_%")), pct(g(bl, "vf_median_%")))
        row("  within 10%", pct(g(ff, "within_10%")), pct(g(bl, "within_10%")))
        row("  mode-ID accuracy", pct(g(ff, "mode_id_acc_%")), "  n/a (scalar)")

    # data efficiency
    de = sorted(root.glob("de_*.json"))
    if de:
        print("\nDATA EFFICIENCY (in-distribution median V_F err vs #train configs)")
        print(f"{'  train fraction':<34}{'FlutterForm':>14}{'MLP baseline':>16}")
        for p in de:
            d = json.loads(p.read_text())
            frac = p.stem.split("_")[1]
            row(f"  frac {frac}", pct(g(d, "flutterform", "vf_median_%")),
                pct(g(d, "baseline", "vf_median_%")))

    print("\n" + "=" * 72)
    print("Reading: the black box wins in-distribution scalar accuracy (easy 6->2\n"
          "regression). FlutterForm's edge is mode-ID (structural — MLP cannot),\n"
          "extrapolation, and predicting the whole flutter diagram, not one number.")
    print("=" * 72)


if __name__ == "__main__":
    main()
