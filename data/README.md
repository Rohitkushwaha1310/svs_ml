# Dataset format and an important limitation

## Required columns

`data/training_dataset.csv` must contain exactly these columns:

| column                  | type   | notes                                        |
|-------------------------|--------|-----------------------------------------------|
| `user_id`                | string | grouping only — never used as a model feature |
| `activity_type`          | string | one of: `mood`, `meditation`, `journal`, `community`, `music` |
| `activity_energy_level`  | number | 0–100, the activity's SVS/energy value        |
| `current_energy_level`   | number | 0–100, the user's energy level BEFORE the activity |
| `predicted_score`        | number | 0–100, the **training target**: the user's REAL, OBSERVED energy level AFTER the activity |

Example row:

```
user_00001, meditation, 80, 55, 63
```

Read as: this user had `current_energy_level = 55`, did a `meditation`
activity with `activity_energy_level = 80`, and their real, observed
post-activity energy score was `predicted_score = 63`.

## ⚠️ Important: `predicted_score` must be a real, observed value

Despite sharing a name with the model's *output*, in the **training CSV**
`predicted_score` must be a genuine historical measurement — e.g. a
self-reported energy check-in the user did shortly after the activity, or
some other ground-truth signal your application already collects.

**Do not train the model on its own past predictions.** If you (or anyone
else) generate this dataset by running an earlier version of the model and
saving its outputs as the new training labels, the model will be learning
to reproduce its own biases and errors rather than anything about real
users. This kind of feedback loop is subtle, doesn't show up as an obvious
bug, and gets worse the longer it runs.

## If you don't yet have enough real labeled data

If your `activity_history` / `wellness_scores` tables don't yet contain
enough (before, activity, after) triples to build a legitimate training
set, do **not** fabricate labels just to fill the gap. Instead:

1. Start collecting the real signal you plan to use as `predicted_score`
   (e.g. a post-activity energy check-in) if you aren't already.
2. Train on a smaller, honest dataset in the meantime — fewer well-labeled
   rows are better than any number of fabricated ones.
3. If you need a placeholder to exercise this codebase end-to-end
   (CI, local development, demos) before real data exists, generate a
   clearly-synthetic dataset and label it as such in any commit message or
   accompanying notes — never mix synthetic and real rows silently in the
   same file.

The `data/training_dataset.csv` shipped alongside this README is
**synthetic**, generated only so the training/evaluation pipeline can be
exercised end-to-end. Replace it with real historical data before training
a model you intend to serve real predictions from.
