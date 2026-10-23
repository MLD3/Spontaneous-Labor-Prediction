"""Evaluate the saved two-stage spontaneous-labor prediction model.

Example
-------
python evaluate_model.py \
  --test-data dummy_test_data.csv \
  --model-dir artifacts \
  --output-dir evaluation_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils import (
    EVENT_COLUMN,
    ID_COLUMN,
    TIME_COLUMN,
    bootstrap_metric_intervals,
    cumulative_incidence_metrics,
    load_dataset,
    model_a_metrics,
    model_b_metrics,
    plot_calibration_summary,
    predict_conditional_cdf,
    save_json,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", default="dummy_test_data.csv", help="Test CSV path.")
    parser.add_argument("--model-dir", default="artifacts", help="Directory containing model_bundle.joblib.")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory for evaluation outputs.")
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[7, 14, 21, 28, 35],
        help="Days after 36+0 at which to evaluate cumulative incidence.",
    )
    return parser.parse_args()


def add_metric_rows(
    rows: list[dict[str, object]],
    model: str,
    scope: str,
    estimates: dict[str, float],
    intervals: dict[str, tuple[float, float]],
    n: int,
    horizon_days: int | None = None,
) -> None:
    for metric, estimate in estimates.items():
        lower, upper = intervals.get(metric, (np.nan, np.nan))
        rows.append(
            {
                "model": model,
                "scope": scope,
                "horizon_days": horizon_days,
                "metric": metric,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": n,
            }
        )


def main() -> None:
    args = parse_args()
    test_data = load_dataset(args.test_data)

    bundle_path = Path(args.model_dir) / "model_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {bundle_path}. Run train_model.py first."
        )
    bundle = joblib.load(bundle_path)

    X_test = transform_features(
        test_data,
        bundle["feature_columns"],
        bundle["imputer"],
        bundle["scaler"],
    )
    true_time = test_data[TIME_COLUMN].to_numpy(dtype=float)
    event_observed = test_data[EVENT_COLUMN].to_numpy(dtype=int)

    X_test_model_a = pd.DataFrame(
        X_test,
        columns=bundle.get(
            "model_a_feature_names",
            [f"Column_{index}" for index in range(X_test.shape[1])],
        ),
    )
    model_a_probability = bundle["model_a"].predict_proba(X_test_model_a)[:, 1]
    raw_model_b_time = bundle["model_b"].predict(X_test)
    model_b_time = bundle["model_b_point_calibrator"].predict(raw_model_b_time)
    model_b_cdf = predict_conditional_cdf(
        model_b_time,
        bundle["cdf_score_cutpoints"],
        bundle["cdf_by_bin"],
    )
    combined_cdf = np.clip(model_a_probability[:, None] * model_b_cdf, 0.0, 1.0)
    days = np.asarray(bundle["cdf_days"], dtype=int)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_frame = pd.DataFrame(
        {
            ID_COLUMN: test_data[ID_COLUMN] if ID_COLUMN in test_data else np.arange(len(test_data)),
            TIME_COLUMN: true_time,
            EVENT_COLUMN: event_observed,
            "model_a_probability": model_a_probability,
            "model_b_predicted_time_days": model_b_time,
        }
    )
    for horizon in args.horizons:
        day_index = int(np.argmin(np.abs(days - horizon)))
        prediction_frame[f"cumulative_incidence_day_{horizon}"] = combined_cdf[:, day_index]
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)

    metric_rows: list[dict[str, object]] = []

    model_a_estimates = model_a_metrics(event_observed, model_a_probability)
    model_a_intervals = bootstrap_metric_intervals(
        len(test_data),
        lambda idx: model_a_metrics(event_observed[idx], model_a_probability[idx]),
        n_bootstraps=args.n_bootstraps,
        random_state=args.random_seed,
    )
    add_metric_rows(
        metric_rows,
        "Model A",
        "all test records",
        model_a_estimates,
        model_a_intervals,
        len(test_data),
    )

    model_b_estimates = model_b_metrics(true_time, model_b_time, event_observed)
    model_b_intervals = bootstrap_metric_intervals(
        len(test_data),
        lambda idx: model_b_metrics(true_time[idx], model_b_time[idx], event_observed[idx]),
        n_bootstraps=args.n_bootstraps,
        random_state=args.random_seed + 1,
    )
    add_metric_rows(
        metric_rows,
        "Model B",
        "observed spontaneous-labor records",
        model_b_estimates,
        model_b_intervals,
        int(event_observed.sum()),
    )

    brier_by_day = []
    baseline_brier_by_day = []
    for horizon in args.horizons:
        day_index = int(np.argmin(np.abs(days - horizon)))
        predicted = combined_cdf[:, day_index]
        observed = ((event_observed == 1) & (true_time <= horizon)).astype(int)
        estimates = cumulative_incidence_metrics(observed, predicted)
        intervals = bootstrap_metric_intervals(
            len(test_data),
            lambda idx, observed=observed, predicted=predicted: cumulative_incidence_metrics(
                observed[idx], predicted[idx]
            ),
            n_bootstraps=args.n_bootstraps,
            random_state=args.random_seed + horizon,
        )
        add_metric_rows(
            metric_rows,
            "Model A + Model B",
            "all test records",
            estimates,
            intervals,
            len(test_data),
            horizon_days=horizon,
        )
        brier_by_day.append(estimates["brier"])
        prevalence = observed.mean()
        baseline_brier_by_day.append(float(np.mean((observed - prevalence) ** 2)))

    integrated_brier = float(np.mean(brier_by_day))
    baseline_integrated_brier = float(np.mean(baseline_brier_by_day))
    brier_skill = (
        1.0 - integrated_brier / baseline_integrated_brier
        if baseline_integrated_brier > 0
        else np.nan
    )
    add_metric_rows(
        metric_rows,
        "Model A + Model B",
        "mean across requested horizons",
        {"mean_brier": integrated_brier, "brier_skill": brier_skill},
        {},
        len(test_data),
    )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)

    plot_calibration_summary(
        event_observed,
        model_a_probability,
        true_time,
        combined_cdf,
        days,
        args.horizons,
        output_dir / "calibration.png",
    )

    summary = {
        "model_bundle": str(bundle_path),
        "test_data": str(args.test_data),
        "n_test": len(test_data),
        "n_observed_spontaneous_labor": int(event_observed.sum()),
        "model_a": model_a_estimates,
        "model_b_observed_only": model_b_estimates,
        "combined_mean_brier": integrated_brier,
        "combined_brier_skill": brier_skill,
        "horizons_days_after_36_weeks": args.horizons,
    }
    save_json(summary, output_dir / "summary.json")

    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
