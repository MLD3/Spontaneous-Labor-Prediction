# Two-Stage Spontaneous-Labor Prediction

This repository is a compact, public-facing implementation of the Model A/Model B workflow used to predict spontaneous labor from features available at 36 weeks and 0 days of gestation.

- **Model A** estimates the probability that spontaneous labor will be observed before a planned delivery intervention.
- **Model B** estimates the number of days from 36+0 weeks to spontaneous labor. It uses a censoring-aware LightGBM objective: observed spontaneous-labor records receive the usual squared-error loss, while a record censored at time `C` is penalized only when the predicted time is earlier than `C`.
- A held-out calibration subset converts Model B point predictions into a conditional cumulative distribution, `P(T <= t | spontaneous labor observed, X)`.
- The final cumulative incidence estimate is:

  `P(spontaneous labor by t | X) = Model A probability × Model B conditional CDF`.

The included CSVs are fully synthetic and contain no patient data.

## Repository files

| File | Purpose |
|---|---|
| `train_model.py` | Loads a training CSV, creates a training/calibration split, fits Models A and B, estimates the Model B conditional CDF, and saves one model bundle. |
| `evaluate_model.py` | Loads the model bundle and test CSV; reports Model A discrimination/calibration, Model B timing accuracy among observed spontaneous-labor records, and combined cumulative-incidence performance. |
| `utils.py` | Shared preprocessing, censoring-aware objective, calibration, metrics, bootstrap intervals, subgroup helpers, labels, and plotting functions consolidated from the notebook utilities. |
| `feature_summary.csv` | Source feature dictionary. `Summary` is the source-cohort prevalence for binary features and the mean for continuous features. |
| `dummy_train_data.csv` | Synthetic training data following the required schema. |
| `dummy_test_data.csv` | Synthetic held-out test data following the required schema. |
| `README.md` | Setup, schema, commands, outputs, and modeling notes. |

## Installation

Python 3.10 or newer is recommended.

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib joblib
```

## Data schema

Each input CSV must contain:

| Column | Meaning |
|---|---|
| `patient_id` | Optional record identifier. It is not used as a model feature. |
| `event_time_days` | Days after 36+0 weeks to spontaneous labor when `event_observed = 1`; otherwise, days to the censoring intervention or censoring-associated admission. |
| `event_observed` | `1` when spontaneous labor time was observed and `0` when the record represents a pregnancy where intervention precedes spontaneous labor. |
| all remaining columns | Numeric model features. Training establishes the feature order. Missing test features are added as missing and median-imputed; extra test columns are ignored. |

The dummy files use 35 readable features selected from `feature_summary.csv`. The scripts also support the full feature matrix as long as it follows the same schema.

## Train the models

From the repository directory:

```bash
python train_model.py \
  --train-data dummy_train_data.csv \
  --model-dir artifacts
```

Important optional arguments:

```text
--validation-fraction 0.20
--n-estimators 200
--learning-rate 0.05
--max-depth 6
--n-jobs 4
--cdf-bins 10
--max-day 49
--random-seed 0
```

Training creates:

```text
artifacts/
├── model_bundle.joblib
└── training_summary.json
```

`model_bundle.joblib` contains the feature order, imputer, scaler, Model A, Model B, point-prediction calibrator, validation-derived score cut points, and conditional CDF table.

## Evaluate the models

```bash
python evaluate_model.py \
  --test-data dummy_test_data.csv \
  --model-dir artifacts \
  --output-dir evaluation_results
```

The default evaluation uses 1,000 test-set bootstrap resamples for percentile intervals. Evaluation creates:

```text
evaluation_results/
├── calibration.png
├── metrics.csv
├── predictions.csv
└── summary.json
```

`metrics.csv` includes:

- Model A AUROC, AUPRC, Brier score, and expected calibration error over the full test population.
- Model B mean absolute error, proportion within 7 and 14 days, and R² among records with observed spontaneous labor.
- Combined cumulative-incidence AUROC, Brier score, and expected calibration error at 7, 14, 21, 28, and 35 days after 36+0 weeks.
- Mean Brier score and Brier skill across the requested horizons.

`predictions.csv` contains Model A probabilities, Model B point predictions, and combined cumulative-incidence predictions at each requested horizon.

## Modeling notes

1. **Calibration data are separate from model fitting.** `train_model.py` makes a held-out calibration split from the supplied training CSV. When repeated `patient_id` values are present, the split is performed at the patient level. Model B point recalibration and score-bin CDF estimation use only this split.
2. **Model B is conditional.** Its direct timing metrics are calculated only where spontaneous labor time is observed. The combined Model A + Model B cumulative incidence is evaluated over all records.
3. **Bootstrap intervals describe test-set sampling uncertainty.** They are percentile intervals from resampling test records with replacement.
4. **External validation is required.** The scripts reproduce the analytical structure but do not establish transportability, clinical utility, or suitability for deployment at another institution.

## Released models

We release Model A and Model B in the root directory. If you are interested in validating the model on your own dataset and would like assistance in porting the model to your setting, please do reach out to the corresponding authors, divyas@cornell.edu and wiensj@umich.edu!

## Using a real dataset

Replace the dummy CSV paths with your prepared train and test files. Keep outcome columns named exactly as specified above. The test feature names should match the training feature names; the evaluation script aligns them automatically and uses the saved training preprocessing objects.
