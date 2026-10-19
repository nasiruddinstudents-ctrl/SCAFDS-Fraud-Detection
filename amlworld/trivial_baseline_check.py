import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

TRAIN_END = "2022-09-07 23:59"
GAP_END = "2022-09-08 23:59"
TEST_END = "2022-09-10 23:59"

df = pd.read_csv("/workspace/HI-Small_Trans.csv")
df["ts"] = pd.to_datetime(df["Timestamp"])
df = df[df["ts"] <= TEST_END].copy()
train = df[df["ts"] <= TRAIN_END].copy()
test = df[df["ts"] > GAP_END].copy()

all_banks = pd.concat([df["From Bank"], df["To Bank"]]).unique()
encoder = LabelEncoder().fit(all_banks)
n_banks = len(encoder.classes_)

train["u"] = encoder.transform(train["From Bank"])
train["v"] = encoder.transform(train["To Bank"])
degree = (train.groupby("u").size().reindex(range(n_banks), fill_value=0) +
          train.groupby("v").size().reindex(range(n_banks), fill_value=0)).values

laund = test[test["Is Laundering"] == 1]
flagged = set(encoder.transform(laund["From Bank"])) | set(encoder.transform(laund["To Bank"]))
y = np.zeros(n_banks)
for b in flagged:
    y[b] = 1.0

touched = set(train["u"]) | set(train["v"])
eval_idx = np.array(sorted(touched))
y_eval = y[eval_idx]
degree_eval = degree[eval_idx]

print("TRIVIAL BASELINE: rank purely by train-window transaction degree, no ML")
print("AUPRC:", average_precision_score(y_eval, degree_eval))
print("AUROC:", roc_auc_score(y_eval, degree_eval))
print("Base rate:", y_eval.mean())
