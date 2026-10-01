"""Train reproducible WasteLens artifacts from the labeled demo dataset."""
import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from wastelens_core import validate_csv, train_models


def train_and_evaluate(path="data/synthetic_historical_data.csv"):
    with open(path, "rb") as source:
        result = validate_csv(source.read())
    if result.errors:
        raise ValueError("Dataset validation failed: " + "; ".join(result.errors))
    models = train_models(result.frame)
    os.makedirs("models", exist_ok=True)
    for key in ["demand_model", "demand_features", "waste_model", "waste_features"]:
        joblib.dump(models[key], os.path.join("models", key + ".joblib"))
    metadata = {"dataset_sha256": result.checksum, "rows": result.valid_rows,
                "date_min": str(models["date_min"].date()), "date_max": str(models["date_max"].date()),
                "algorithms": {"demand": models["demand_algorithm"], "waste": models["waste_algorithm"]},
                "model_version": f"demand-{models['demand_algorithm']}_waste-{models['waste_algorithm']}-v2-{result.checksum[:8]}",
                "quality": models["quality"], "demand_metrics": models["demand_metrics"],
                "waste_metrics": models["waste_metrics"],
                "library_versions": {"python": __import__("sys").version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}}
    with open("models/model_metadata.json", "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
    print(json.dumps(metadata, indent=2))
    return models


if __name__ == "__main__":
    train_and_evaluate()
