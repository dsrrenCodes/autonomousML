from pathlib import Path

from autogluon.tabular import TabularDataset, TabularPredictor

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "adversarial_suite"

data = TabularDataset(str(DATA_DIR / "titanic_leakage.csv"))

predictor = TabularPredictor(label="Survived").fit(
    data, time_limit=60, presets="medium_quality"
)

print("Problem type:", predictor.problem_type)
print(predictor.leaderboard())