"""
SVS (Subjective Vitality Score) synthetic training dataset generator.

Run:
    pip install numpy
    python generate_svs_dataset.py

Output:
    training_dataset.csv  (in the same folder as this script)

Adjust N_USERS / MIN_ACT / MAX_ACT below to control final dataset size.
With defaults (N_USERS=46000, 4-9 activities/user) you'll get
roughly 290,000-300,000 rows, ~15% of which are first-activity
(current_energy_level = 0) rows.
"""

import csv
import numpy as np

# ----------------------------------------------------------------------
# CONFIG - tweak these to change dataset size
# ----------------------------------------------------------------------
SEED = 42
N_USERS = 46000          # unique users (at least 20,000 required)
MIN_ACT = 4              # min activities per user
MAX_ACT = 9              # max activities per user (inclusive)
OUTPUT_PATH = "training_dataset.csv"

ACTIVITY_TYPES = ["mood", "meditation", "journal", "community", "music"]

# Moderate, mildly-to-moderately positive per-activity effect ranges
# (only applied on activities after the first one)
ACTIVITY_EFFECT_RANGE = {
    "mood": (-1.0, 4.0),
    "meditation": (1.0, 6.0),
    "journal": (0.5, 5.0),
    "community": (0.5, 5.0),
    "music": (-1.0, 4.0),
}

# ----------------------------------------------------------------------

rng = np.random.default_rng(SEED)


def sample_energy_level(n):
    """Realistic, non-uniform distribution across 0-100."""
    bucket_edges = [
        (0, 20, 0.12),
        (20, 40, 0.16),
        (40, 60, 0.24),
        (60, 80, 0.26),
        (80, 100, 0.22),
    ]
    buckets = rng.choice(len(bucket_edges), size=n, p=[b[2] for b in bucket_edges])
    lows = np.array([bucket_edges[b][0] for b in buckets], dtype=float)
    highs = np.array([bucket_edges[b][1] for b in buckets], dtype=float)
    vals = rng.uniform(lows, highs)
    return np.round(vals, 1)


def generate_rows():
    rows = []
    for u in range(N_USERS):
        user_id = f"user_{u + 1:06d}"

        n_activities = int(rng.integers(MIN_ACT, MAX_ACT + 1))
        act_types = rng.choice(ACTIVITY_TYPES, size=n_activities)
        act_energies = sample_energy_level(n_activities)

        current = 0.0
        for i in range(n_activities):
            a_type = act_types[i]
            a_energy = float(act_energies[i])

            if i == 0:
                # FIRST ACTIVITY: predicted_score ~= activity_energy_level
                current_energy = 0.0
                noise = float(np.clip(rng.normal(0, 1.8), -5, 5))
                predicted = a_energy + noise
            else:
                # SUBSEQUENT ACTIVITY: blend of current + activity + small effect/noise
                current_energy = current
                low, high = ACTIVITY_EFFECT_RANGE[a_type]
                effect = rng.uniform(low, high)
                noise = rng.normal(0, 2.5)
                base = 0.60 * current_energy + 0.40 * a_energy
                predicted = base + effect + noise

            predicted = round(float(np.clip(predicted, 0, 100)), 1)

            rows.append(
                (user_id, a_type, round(a_energy, 1), round(current_energy, 1), predicted)
            )
            current = predicted

    return rows


def dedupe(rows):
    """Guarantee no exact duplicate rows by nudging predicted_score if needed."""
    seen = set()
    out = []
    for r in rows:
        while r in seen:
            user_id, a_type, a_e, c_e, pred = r
            pred = round(min(100.0, max(0.0, float(pred) + 0.1)), 1)
            r = (user_id, a_type, a_e, c_e, pred)
        seen.add(r)
        out.append(r)
    return out


def main():
    rows = generate_rows()
    rows = dedupe(rows)

    total = len(rows)
    zero_count = sum(1 for r in rows if r[3] == 0.0)
    uniq_users = len(set(r[0] for r in rows))

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["user_id", "activity_type", "activity_energy_level", "current_energy_level", "predicted_score"]
        )
        writer.writerows(rows)

    print(f"Wrote {total} rows to {OUTPUT_PATH}")
    print(f"Unique users: {uniq_users}")
    print(f"Rows with current_energy_level = 0: {zero_count} ({zero_count / total * 100:.2f}%)")


if __name__ == "__main__":
    main()