# SVS ML

A machine-learning microservice that predicts a wellness app user's
post-activity energy score, plus a webhook integration that connects it to
Supabase.

---

## Architecture

```
Supabase (activity_history INSERT)
        │  Database Webhook
        ▼
POST /webhook/activity  (api/webhook.py)
        │  1. reads wellness_scores.final_energy_level for this user
        │  2. maps Supabase fields -> ML fields
        ▼
POST /predict  (api/main.py, over HTTP via ML_MODEL_URL)
        │  returns {"predicted_score": ...}
        ▼
back in the webhook:
        UPDATE wellness_scores
        SET final_energy_level = predicted_score
        WHERE user_id = activity.user_id
        │
        UPDATE activity_history
        SET process = 'done'
        WHERE id = activity.id
```

The webhook calls the ML API over **HTTP**, not by importing the model
directly. That's deliberate: today both live in the same process
(`ML_MODEL_URL=http://127.0.0.1:8000/predict`), but you can deploy the ML
API as its own service later and just repoint `ML_MODEL_URL` — no code
changes needed on either side.

Training and prediction are fully separate: `training/train.py` builds and
saves `models/svs_model.pkl` offline; the API only ever loads that file and
predicts. Nothing here ever retrains on request.

## Supabase schema (source of truth — never renamed by this project)

```
activity_history                      wellness_scores
-----------------                     ----------------
id               uuid                 id                  uuid
user_id          uuid                 user_id             uuid
activity_type    text                 final_energy_level  int4
title            text                 breakdown           jsonb
subtitle         text (nullable)      computed_at         timestamptz
metadata         jsonb                created_at          timestamptz
created_at       timestamptz
energy_level     int4 (nullable)   <- activity's SVS/energy value
process          text (nullable)  <- NULL | "done"  (idempotency flag)
```

## Field mapping

```
SUPABASE                              ML MODEL INPUT              MODEL OUTPUT       APPLICATION
--------                              ---------------              ------------       -----------
activity_history.activity_type  ───►  activity_type
activity_history.energy_level   ───►  activity_energy_level
wellness_scores.final_energy_level ─► current_energy_level
                                                          ────────► predicted_score ─► wellness_scores.final_energy_level
                                                                                        (written back, same user_id)
```

`predicted_score` is a pure ML/backend name — it is **never** a Supabase
column. `wellness_scores.final_energy_level` is both where
`current_energy_level` is read from and where the new `predicted_score` is
written back to. This mapping happens explicitly, once, in
`api/webhook.py` — nowhere else.

## Project structure

```
project/
├── api/
│   ├── __init__.py
│   ├── main.py             # FastAPI app: GET /health, POST /predict, mounts webhook router
│   ├── webhook.py          # POST /webhook/activity
│   ├── supabase_client.py  # the only module that talks to Supabase
│   └── ml_client.py        # HTTP client that calls ML_MODEL_URL
├── preprocessing/
│   ├── __init__.py
│   └── preprocess.py       # single source of truth: feature/target names + pipeline
├── training/
│   ├── __init__.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── training_dataset.csv   # SYNTHETIC example data — see data/README.md
│   └── README.md              # dataset format + an important warning
├── models/
│   ├── svs_model.pkl
│   ├── model_metrics.json
│   └── test_split.csv
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_api.py
│   └── test_webhook.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:
- macOS / Linux: `source .venv/bin/activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`

## 2. Install requirements

```bash
pip install -r requirements.txt
```

XGBoost is optional — uncomment the line in `requirements.txt` if you want
it included as an extra candidate model during training.

## 3. Create `.env`

```bash
cp .env.example .env
```

Fill in:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
ML_MODEL_URL=http://127.0.0.1:8000/predict
WEBHOOK_SECRET=some-long-random-string
```

Never commit the real `.env` — `.gitignore` already excludes it.
`SUPABASE_SERVICE_ROLE_KEY` must never be shipped to any frontend or mobile
client.

## 4. Train the model

```bash
python training/train.py
```

⚠️ Before training on anything beyond the bundled synthetic example, read
`data/README.md` — specifically, do not train on a model's own past
predictions. The training target (`predicted_score` in the CSV) must be a
real, observed post-activity energy value.

This validates the dataset, splits users into train/validation/test
(no user appears in more than one split), compares several regressors,
picks the best on validation MAE, retrains it on train+validation, and
saves:

- `models/svs_model.pkl` — the fitted preprocessing + model pipeline
- `models/model_metrics.json` — full performance report
- `models/test_split.csv` — the held-out test rows, for `evaluate.py`

## 5. Evaluate the model

```bash
python training/evaluate.py
```

Loads the saved model and the saved test split, prints MAE / RMSE / R², and
a handful of actual-vs-predicted example rows. Does not retrain anything.

## 6. Start the FastAPI service

```bash
uvicorn api.main:app --reload
```

Serves on `http://127.0.0.1:8000` by default.

## 7. Test `/health`

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

## 8. Test `/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "activity_type": "meditation",
    "activity_energy_level": 80,
    "current_energy_level": 45
  }'
```

```json
{"predicted_score": 62.31}
```

Note the field is `predicted_score`, not `energy_level`.

## 9. Start ngrok (for local webhook testing)

With the API already running on port 8000 in one terminal, in another:

```bash
ngrok http 8000
```

ngrok prints a public URL like `https://abc123.ngrok-free.app`. **This
domain changes every time you restart ngrok** unless you're on a paid plan
with a reserved/static domain — always use whatever URL ngrok gives you
*right now*, not one from a previous session.

## 10. Configure the Supabase Database Webhook

In the Supabase dashboard, under Database → Webhooks:

| Setting | Value |
|---|---|
| Table | `activity_history` |
| Event | `INSERT` |
| Type | HTTP Request |
| Method | `POST` |
| URL | `https://<CURRENT-NGROK-DOMAIN>/webhook/activity` |
| Header | `Content-Type: application/json` |
| Header | `X-Webhook-Secret: <same value as WEBHOOK_SECRET in your .env>` |

`activity_history.id` and `activity_history.user_id` are real Postgres
`uuid` columns — the webhook validates both as UUIDs and returns `400` for
anything else, before ever touching Supabase or the model.

## 11. Test an actual activity insert

Insert a row into `activity_history` for a `user_id` that already has a
row in `wellness_scores` (via the Supabase Table Editor, SQL editor, or
your app itself):

```sql
insert into activity_history (user_id, activity_type, title, energy_level)
values ('<a real user_id uuid>', 'meditation', 'Evening meditation', 50);
```

## 12. Verify `wellness_scores.final_energy_level` changed

```sql
select user_id, final_energy_level, computed_at
from wellness_scores
where user_id = '<the same user_id>'
order by computed_at desc
limit 1;
```

It should now reflect the model's prediction. Also check:

```sql
select id, process from activity_history where id = '<the activity id>';
```

`process` should now read `'done'`. Re-delivering the same webhook (or
Supabase retrying it) for that same activity `id` will get a `409
Conflict` and will not change `final_energy_level` again.

---

## Idempotency / duplicate webhook handling

Supabase (and HTTP delivery in general) can redeliver the same webhook.
Applying the same activity's prediction twice would shift a user's energy
level twice for one real event.

**What's implemented:** the existing `activity_history.process` column is
used directly — no new table.

- Before predicting, the webhook re-reads `process` for that activity's
  `id` **fresh from the database** — never from the webhook payload
  itself, since a redelivered webhook still carries the row exactly as it
  looked at INSERT time (`process = NULL`), even after a prior delivery
  already set it to `"done"`.
- If `process` is not `NULL` → `409 Conflict`, nothing else happens.
- If `process IS NULL` → proceed, then set `process = "done"` for that
  activity's `id` once the prediction and the `wellness_scores` update
  both succeed.

If the ML API call fails, `wellness_scores.final_energy_level` is **not**
touched and `process` stays `NULL`, so a legitimate retry can still
succeed later.

## Error handling reference

`POST /webhook/activity` returns:

| Situation | Status |
|---|---|
| Missing/invalid `X-Webhook-Secret` | 401 |
| Malformed JSON body | 400 |
| Missing `user_id`, `activity_type`, or a non-UUID `id`/`user_id` | 400 |
| Invalid `activity_type` | 400 |
| Wrong table or non-INSERT event | 400 |
| `activity_energy_level` cannot be determined (energy_level NULL, no usable metadata) | 400 |
| Duplicate `activity_id` (`process` already set) | 409 |
| Activity id not found in `activity_history` | 404 |
| No `wellness_scores` row for that `user_id` | 404 |
| ML API unreachable, timed out, or returned an error | 502 |

Nothing fails silently — every rejection returns a JSON `detail` message
and is logged. Webhook payloads are logged with any field named like a
secret/token/credential automatically redacted.

## Running the tests

```bash
pytest tests/ -v
```

- `test_preprocessing.py` and `test_model.py` only need `pandas`/`scikit-learn`/`joblib`.
- `test_api.py` and `test_webhook.py` need a trained model (`python training/train.py` first)
  because the API loads `models/svs_model.pkl` on startup.
- `test_webhook.py` mocks all Supabase calls and the ML API HTTP call — no
  real credentials or network access needed.
