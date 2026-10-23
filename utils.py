"""Shared utilities for the two-stage spontaneous-labor prediction pipeline.

The public pipeline mirrors the analysis notebook:

* Model A estimates the probability that spontaneous labor is observed before
  a planned delivery intervention.
* Model B estimates time from 36+0 weeks to spontaneous labor. For censored
  records, the custom loss penalizes predictions earlier than the censoring
  time but does not penalize later predictions.
* Validation-set Model B scores are converted to a conditional cumulative
  distribution function (CDF), which is multiplied by Model A probabilities
  to obtain a cumulative incidence function over the full population.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

TIME_COLUMN = "event_time_days"
EVENT_COLUMN = "event_observed"
ID_COLUMN = "patient_id"
OUTCOME_COLUMNS = {TIME_COLUMN, EVENT_COLUMN, ID_COLUMN}

# Labels consolidated from the original figure utility module.
PRETTY_METHOD = {
    "model_a": "Model A",
    "model_b": "Model B",
    "combined": "Model A + Model B",
    "edd": "EDD (Current Approach)",
    "median": "Median",
}

PRETTY_SUBGROUP = {
    "overall": "Overall",
    "BMI>40": "BMI > 40",
    "Multiple gestations": "Multiple gestations",
    "Gestational Diabetes": "Gestational diabetes",
    "Gestational Hypertension": "Gestational hypertension",
    "High risk pregnancy": "High-risk pregnancy",
    "prior_pregnancy": "Prior pregnancy",
    "no_prior_pregnancy": "No prior pregnancy",
    "any_PE_CH_GH": "Hypertensive disorder",
}


class CensoringAwareObjective:
    """LightGBM objective for right-censored time-to-event regression.

    For observed spontaneous-labor events, squared error is always used. For a
    record censored at time C, squared error is used only when the model predicts
    a time earlier than C. Predictions at or after C are compatible with the
    observed censoring information and receive an approximately zero gradient.
    """

    def __init__(self, event_observed: Sequence[int | bool]) -> None:
        self.event_observed = np.asarray(event_observed, dtype=bool)

    def __call__(self, predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        true_time = np.asarray(dataset.get_label(), dtype=float)
        if len(true_time) != len(self.event_observed):
            raise ValueError(
                "The censoring indicator must have the same length as the "
                "LightGBM training dataset."
            )

        residual = predictions - true_time
        use_loss = self.event_observed | ((~self.event_observed) & (predictions < true_time))
        gradient = np.where(use_loss, 2.0 * residual, 1e-6)
        hessian = np.where(use_loss, 2.0, 1e-6)
        return gradient, hessian


class IdentityCalibrator:
    """Fallback calibrator with the same predict interface as scikit-learn."""

    def predict(self, values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=float)


class ConstantCalibrator:
    """Constant-probability fallback used when an outcome has one class."""

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict(self, values: Sequence[float]) -> np.ndarray:
        values = np.asarray(values)
        return np.full(values.shape[0], self.probability, dtype=float)


def load_dataset(path: str | Path, require_outcomes: bool = True) -> pd.DataFrame:
    """Load a CSV and validate the public repository schema."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    if require_outcomes:
        missing = {TIME_COLUMN, EVENT_COLUMN} - set(data.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        if data[TIME_COLUMN].isna().any():
            raise ValueError(f"{TIME_COLUMN} contains missing values.")

        invalid_event = ~data[EVENT_COLUMN].isin([0, 1, False, True])
        if invalid_event.any():
            raise ValueError(f"{EVENT_COLUMN} must contain only 0/1 values.")

        if (data[TIME_COLUMN] < 0).any():
            raise ValueError(f"{TIME_COLUMN} must be nonnegative.")

    return data


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding identifiers and outcomes."""

    feature_columns = [column for column in data.columns if column not in OUTCOME_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns were found.")

    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(data[column])]
    if non_numeric:
        raise TypeError(
            "All model features must be numeric. Non-numeric columns: "
            + ", ".join(non_numeric[:10])
        )
    return feature_columns


def fit_preprocessor(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    """Median-impute and standardize the feature matrix."""

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(data.loc[:, feature_columns])
    transformed = scaler.fit_transform(imputed)
    return transformed, imputer, scaler


def transform_features(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    """Align a dataset to the training feature order and apply preprocessing."""

    aligned = data.copy()
    missing = [column for column in feature_columns if column not in aligned.columns]
    for column in missing:
        aligned[column] = np.nan

    matrix = aligned.loc[:, feature_columns]
    imputed = imputer.transform(matrix)
    return scaler.transform(imputed)


def fit_point_calibrator(
    raw_scores: Sequence[float],
    true_times: Sequence[float],
) -> IdentityCalibrator | IsotonicRegression:
    """Fit the notebook's monotone point-prediction recalibration."""

    raw_scores = np.asarray(raw_scores, dtype=float)
    true_times = np.asarray(true_times, dtype=float)
    valid = np.isfinite(raw_scores) & np.isfinite(true_times)
    if valid.sum() < 2 or np.unique(raw_scores[valid]).size < 2:
        return IdentityCalibrator()

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_scores[valid], true_times[valid])
    return calibrator


def fit_conditional_cdf(
    calibrated_scores: Sequence[float],
    observed_times: Sequence[float],
    n_score_bins: int = 10,
    max_day: int = 49,
) -> dict[str, np.ndarray]:
    """Estimate P(T <= t | spontaneous labor observed, Model B score bin).

    Quantile cut points are estimated on observed validation cases. Within each
    score bin, the empirical CDF of observed event times is calculated for days
    1 through ``max_day`` after 36+0 weeks.
    """

    scores = np.asarray(calibrated_scores, dtype=float)
    times = np.floor(np.asarray(observed_times, dtype=float)).astype(int)
    valid = np.isfinite(scores) & np.isfinite(times)
    scores = scores[valid]
    times = times[valid]

    if len(scores) == 0:
        raise ValueError("At least one observed validation event is required for CDF calibration.")

    requested_bins = max(1, min(int(n_score_bins), len(scores)))
    if np.unique(scores).size == 1:
        score_edges = np.array([-np.inf, np.inf], dtype=float)
        score_cutpoints = np.array([], dtype=float)
        score_bin = np.zeros(len(scores), dtype=int)
    else:
        _, finite_edges = pd.qcut(
            scores,
            q=requested_bins,
            retbins=True,
            duplicates="drop",
        )
        score_cutpoints = np.asarray(finite_edges[1:-1], dtype=float)
        score_edges = np.concatenate(([-np.inf], score_cutpoints, [np.inf]))
        score_bin = np.digitize(scores, score_cutpoints)

    days = np.arange(1, int(max_day) + 1)
    n_actual_bins = len(score_edges) - 1
    cdf_by_bin = np.zeros((n_actual_bins, len(days)), dtype=float)

    global_cdf = np.array([np.mean(times <= day) for day in days], dtype=float)
    for bin_id in range(n_actual_bins):
        bin_times = times[score_bin == bin_id]
        if len(bin_times) == 0:
            cdf_by_bin[bin_id] = global_cdf
        else:
            cdf_by_bin[bin_id] = [np.mean(bin_times <= day) for day in days]

    cdf_by_bin = np.maximum.accumulate(cdf_by_bin, axis=1)
    return {
        "days": days,
        "score_cutpoints": score_cutpoints,
        "cdf_by_bin": np.clip(cdf_by_bin, 0.0, 1.0),
    }


def predict_conditional_cdf(
    calibrated_scores: Sequence[float],
    score_cutpoints: Sequence[float],
    cdf_by_bin: np.ndarray,
) -> np.ndarray:
    """Assign Model B scores to validation-derived CDF bins."""

    scores = np.asarray(calibrated_scores, dtype=float)
    score_bin = np.digitize(scores, np.asarray(score_cutpoints, dtype=float))
    score_bin = np.clip(score_bin, 0, cdf_by_bin.shape[0] - 1)
    return np.asarray(cdf_by_bin, dtype=float)[score_bin]


def expected_calibration_error(
    y_true: Sequence[int | bool],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected calibration error using quantile-based probability bins."""

    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(probabilities)
    y_true = y_true[valid]
    probabilities = probabilities[valid]
    if len(y_true) == 0:
        return float("nan")

    try:
        bin_ids = pd.qcut(
            probabilities,
            q=min(n_bins, len(probabilities)),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return float(np.abs(y_true.mean() - probabilities.mean()))

    bin_ids = np.asarray(bin_ids, dtype=float)
    valid_bins = ~np.isnan(bin_ids)
    if not valid_bins.any():
        return float(np.abs(y_true.mean() - probabilities.mean()))

    y_true = y_true[valid_bins]
    probabilities = probabilities[valid_bins]
    bin_ids = bin_ids[valid_bins].astype(int)

    ece = 0.0
    for bin_id in np.unique(bin_ids):
        mask = bin_ids == bin_id
        ece += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(ece)


def quantile_calibration_curve(
    y_true: Sequence[int | bool],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return mean predicted and observed probabilities by quantile bin."""

    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(probabilities)
    y_true = y_true[valid]
    probabilities = probabilities[valid]
    if len(y_true) == 0:
        return pd.DataFrame(columns=["mean_predicted", "observed", "n"])

    try:
        bins = pd.qcut(
            probabilities,
            q=min(n_bins, len(probabilities)),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        bins = np.zeros(len(probabilities), dtype=int)

    frame = pd.DataFrame({"y": y_true, "p": probabilities, "bin": bins}).dropna()
    return (
        frame.groupby("bin", observed=True)
        .agg(mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )


def safe_roc_auc(y_true: Sequence[int | bool], probabilities: Sequence[float]) -> float:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def safe_average_precision(y_true: Sequence[int | bool], probabilities: Sequence[float]) -> float:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, probabilities))


def model_a_metrics(y_true: Sequence[int | bool], probabilities: Sequence[float]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "auroc": safe_roc_auc(y_true, probabilities),
        "auprc": safe_average_precision(y_true, probabilities),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
    }


def model_b_metrics(
    true_time: Sequence[float],
    predicted_time: Sequence[float],
    event_observed: Sequence[int | bool],
) -> dict[str, float]:
    true_time = np.asarray(true_time, dtype=float)
    predicted_time = np.asarray(predicted_time, dtype=float)
    observed = np.asarray(event_observed, dtype=bool)
    if observed.sum() == 0:
        return {"mae_days": np.nan, "within_7_days": np.nan, "within_14_days": np.nan, "r2": np.nan}

    errors = np.abs(true_time[observed] - predicted_time[observed])
    return {
        "mae_days": float(mean_absolute_error(true_time[observed], predicted_time[observed])),
        "within_7_days": float(np.mean(errors <= 7.0)),
        "within_14_days": float(np.mean(errors <= 14.0)),
        "r2": float(r2_score(true_time[observed], predicted_time[observed])) if observed.sum() >= 2 else np.nan,
    }


def cumulative_incidence_metrics(
    observed_event_by_horizon: Sequence[int | bool],
    predicted_probability: Sequence[float],
) -> dict[str, float]:
    y_true = np.asarray(observed_event_by_horizon, dtype=int)
    probabilities = np.asarray(predicted_probability, dtype=float)
    return {
        "auroc": safe_roc_auc(y_true, probabilities),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
    }


def bootstrap_metric_intervals(
    n_samples: int,
    metric_function: Callable[[np.ndarray], Mapping[str, float]],
    n_bootstraps: int = 1000,
    random_state: int = 0,
) -> dict[str, tuple[float, float]]:
    """Compute percentile intervals from test-set bootstrap resamples."""

    if n_bootstraps <= 0:
        return {}

    rng = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {}
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        result = metric_function(indices)
        for name, value in result.items():
            values.setdefault(name, []).append(float(value))

    intervals: dict[str, tuple[float, float]] = {}
    for name, metric_values in values.items():
        array = np.asarray(metric_values, dtype=float)
        if np.isfinite(array).any():
            lower, upper = np.nanpercentile(array, [2.5, 97.5])
            intervals[name] = (float(lower), float(upper))
        else:
            intervals[name] = (np.nan, np.nan)
    return intervals


def get_subgroup_mask(data: pd.DataFrame, subgroup_name: str) -> np.ndarray:
    """Create common subgroup masks from readable feature names.

    The helper uses case-insensitive feature-name matching so it can operate on
    the readable EHR feature names used in ``feature_summary.csv``.
    """

    lower_names = {column: column.lower() for column in data.columns}

    def any_positive(terms: Iterable[str]) -> np.ndarray:
        columns = [
            column
            for column, lower in lower_names.items()
            if any(term.lower() in lower for term in terms) and "missing" not in lower
        ]
        if not columns:
            return np.zeros(len(data), dtype=bool)
        return data.loc[:, columns].fillna(0).to_numpy().sum(axis=1) != 0

    if subgroup_name == "BMI>40":
        numeric_bmi = [column for column, lower in lower_names.items() if lower in {"bmi", "bmi_value"}]
        if numeric_bmi:
            return data[numeric_bmi[0]].to_numpy() > 40
        return any_positive(["bmi [value from 35.85"])
    if subgroup_name == "Multiple gestations":
        return any_positive(["multiple gestation"])
    if subgroup_name == "Gestational Diabetes":
        return any_positive(["diabetes in pregnancy"])
    if subgroup_name == "Gestational Hypertension":
        return any_positive(["gestational hypertension"])
    if subgroup_name == "High risk pregnancy":
        return any_positive(["high risk pregnancy"])
    if subgroup_name == "prior_pregnancy":
        return any_positive(["mean length of previous pregnancies", "number of total pregnancies"])
    if subgroup_name == "no_prior_pregnancy":
        return ~get_subgroup_mask(data, "prior_pregnancy")
    if subgroup_name == "any_PE_CH_GH":
        return any_positive(["pre-eclampsia", "hypertension"])
    raise ValueError(f"Unsupported subgroup: {subgroup_name}")


def plot_calibration_summary(
    event_observed: Sequence[int | bool],
    model_a_probability: Sequence[float],
    true_time: Sequence[float],
    combined_cdf: np.ndarray,
    days: Sequence[int],
    horizons: Sequence[int],
    output_path: str | Path,
) -> None:
    """Save Model A and combined-CDF calibration plots."""

    event_observed = np.asarray(event_observed, dtype=int)
    true_time = np.asarray(true_time, dtype=float)
    days = np.asarray(days, dtype=int)
    horizons = list(horizons)

    n_panels = 1 + len(horizons)
    figure, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.0), squeeze=False)
    axes = axes[0]

    curve = quantile_calibration_curve(event_observed, model_a_probability)
    axes[0].plot(curve["mean_predicted"], curve["observed"], marker="o")
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_title("Model A")

    for axis, horizon in zip(axes[1:], horizons):
        day_index = int(np.argmin(np.abs(days - horizon)))
        observed = ((event_observed == 1) & (true_time <= horizon)).astype(int)
        predicted = combined_cdf[:, day_index]
        curve = quantile_calibration_curve(observed, predicted)
        axis.plot(curve["mean_predicted"], curve["observed"], marker="o")
        axis.plot([0, 1], [0, 1], linestyle="--")
        axis.set_title(f"By day {horizon}")

    for axis in axes:
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed proportion")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_json(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write JSON while converting NumPy scalar values."""

    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2, sort_keys=True)
