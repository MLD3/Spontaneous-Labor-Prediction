"""Train the two-stage spontaneous-labor prediction model.

Example
-------
python train_model.py --train-data dummy_train_data.csv --model-dir artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from utils import (
    CensoringAwareObjective,
    EVENT_COLUMN,
    ID_COLUMN,
    TIME_COLUMN,
    fit_conditional_cdf,
    fit_point_calibrator,
    fit_preprocessor,
    get_feature_columns,
    load_dataset,
    model_a_metrics,
    model_b_metrics,
    save_json,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="dummy_train_data.csv", help="Training CSV path.")
    parser.add_argument("--model-dir", default="artifacts", help="Directory for trained artifacts.")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=4, help="LightGBM worker threads.")
    parser.add_argument("--cdf-bins", type=int, default=10)
    parser.add_argument("--max-day", type=int, default=49)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1.")

    data = load_dataset(args.train_data)
    feature_columns = get_feature_columns(data)
    event_observed = data[EVENT_COLUMN].astype(int).to_numpy()

    row_indices = np.arange(len(data))
    has_repeated_patients = (
        ID_COLUMN in data.columns
        and data[ID_COLUMN].nunique(dropna=False) < len(data)
    )
    if has_repeated_patients:
        group_split = GroupShuffleSplit(
            n_splits=1,
            test_size=args.validation_fraction,
            random_state=args.random_seed,
        )
        train_indices, validation_indices = next(
            group_split.split(row_indices, groups=data[ID_COLUMN])
        )
    else:
        stratify = event_observed if np.unique(event_observed).size == 2 else None
        train_indices, validation_indices = train_test_split(
            row_indices,
            test_size=args.validation_fraction,
            random_state=args.random_seed,
            stratify=stratify,
        )

    train_data = data.iloc[train_indices].reset_index(drop=True)
    validation_data = data.iloc[validation_indices].reset_index(drop=True)

    X_train, imputer, scaler = fit_preprocessor(train_data, feature_columns)
    X_validation = transform_features(validation_data, feature_columns, imputer, scaler)

    y_train = train_data[TIME_COLUMN].to_numpy(dtype=float)
    event_train = train_data[EVENT_COLUMN].to_numpy(dtype=int)
    y_validation = validation_data[TIME_COLUMN].to_numpy(dtype=float)
    event_validation = validation_data[EVENT_COLUMN].to_numpy(dtype=int)

    common_parameters = {
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": min(31, 2 ** args.max_depth - 1) if args.max_depth > 0 else 31,
        "verbosity": -1,
        "seed": args.random_seed,
        "feature_fraction_seed": args.random_seed,
        "bagging_seed": args.random_seed,
        "num_threads": args.n_jobs,
        "force_col_wise": True,
    }

    # Model A: probability that spontaneous labor is observed before intervention.
    model_a_feature_names = [f"Column_{index}" for index in range(X_train.shape[1])]
    X_train_model_a = pd.DataFrame(X_train, columns=model_a_feature_names)
    X_validation_model_a = pd.DataFrame(X_validation, columns=model_a_feature_names)
    model_a = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=common_parameters["num_leaves"],
        n_jobs=args.n_jobs,
        force_col_wise=True,
        random_state=args.random_seed,
        verbosity=-1,
    )
    model_a.fit(X_train_model_a, event_train)

    # Model B: censoring-aware regression for time from 36+0 to spontaneous labor.
    model_b_dataset = lgb.Dataset(X_train, label=y_train)
    model_b_parameters = dict(common_parameters)
    model_b_parameters["objective"] = CensoringAwareObjective(event_train)
    model_b = lgb.train(
        model_b_parameters,
        model_b_dataset,
        num_boost_round=args.n_estimators,
    )

    raw_validation_time = model_b.predict(X_validation)
    observed_validation = event_validation == 1
    point_calibrator = fit_point_calibrator(
        raw_validation_time[observed_validation],
        y_validation[observed_validation],
    )
    calibrated_validation_time = point_calibrator.predict(raw_validation_time)

    cdf_calibration = fit_conditional_cdf(
        calibrated_validation_time[observed_validation],
        y_validation[observed_validation],
        n_score_bins=args.cdf_bins,
        max_day=args.max_day,
    )

    bundle = {
        "schema_version": 1,
        "feature_columns": feature_columns,
        "imputer": imputer,
        "scaler": scaler,
        "model_a": model_a,
        "model_a_feature_names": model_a_feature_names,
        "model_b": model_b,
        "model_b_point_calibrator": point_calibrator,
        "cdf_days": cdf_calibration["days"],
        "cdf_score_cutpoints": cdf_calibration["score_cutpoints"],
        "cdf_by_bin": cdf_calibration["cdf_by_bin"],
        "time_column": TIME_COLUMN,
        "event_column": EVENT_COLUMN,
        "id_column": ID_COLUMN,
        "training_arguments": vars(args),
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    validation_a_probability = model_a.predict_proba(X_validation_model_a)[:, 1]
    validation_b_metrics = model_b_metrics(
        y_validation,
        calibrated_validation_time,
        event_validation,
    )
    validation_summary = {
        "n_total": len(data),
        "n_train": len(train_data),
        "n_validation": len(validation_data),
        "n_features": len(feature_columns),
        "training_event_fraction": float(event_train.mean()),
        "validation_event_fraction": float(event_validation.mean()),
        "model_a_validation": model_a_metrics(event_validation, validation_a_probability),
        "model_b_validation_observed_only": validation_b_metrics,
        "model_bundle": str(bundle_path),
    }
    save_json(validation_summary, model_dir / "training_summary.json")

    print(f"Saved model bundle: {bundle_path}")
    print(f"Features: {len(feature_columns):,}")
    print(f"Training records: {len(train_data):,}; validation records: {len(validation_data):,}")
    if np.unique(event_validation).size == 2:
        print(f"Model A validation AUROC: {roc_auc_score(event_validation, validation_a_probability):.3f}")
    print(f"Model B validation MAE among observed events: {validation_b_metrics['mae_days']:.3f} days")


if __name__ == "__main__":
    main()
