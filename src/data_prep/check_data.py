"""python -m src.data_prep.check_data  -> split/leak integrity report (run before every experiment)."""
import json
from src.data_prep.loaders import integrity_report
if __name__ == "__main__":
    r = integrity_report(); print(json.dumps(r, indent=2, default=str))
    bad = r["registry_not_in_split"] + r["cards_not_in_registry"] + sum(v["target_in_wrong_split"] + v["target_not_in_split_map"] for v in r["query_leaks"].values())
    print("\nPASS" if bad == 0 else f"\nPROBLEMS FOUND ({bad}) - fix before training")
