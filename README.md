# Automated Accident Detector (Deep Learning, CNN+LSTM + Motion Heuristic + Firebase Auth)

```
accident_detector/
├── config.py
├── data/
│   ├── preprocessing.py
│   └── dataset.py
├── models/
│   └── model.py                    # MODEL: ResNet18 CNN + LSTM
├── controllers/
│   ├── controller.py                # CONTROLLER: training/inference + ensemble
│   └── auth_controller.py           # CONTROLLER: Firebase auth wrapper
├── views/
│   └── app.py                        # VIEW: Streamlit dashboard (login-gated)
├── utils/
│   └── motion_heuristic.py          # training-free optical-flow accident signal
├── .streamlit/
│   └── secrets.toml.example         # template — copy to secrets.toml locally
├── train.py
├── predict.py
├── make_dummy_data.py
└── checkpoints/
```

## What's new in this version

### 1. Motion-heuristic ensemble (real-world robustness without extra training)

The CNN+LSTM model's accuracy is capped by how much real labeled data it's
seen. To make the system meaningfully useful on **real** footage right away,
`utils/motion_heuristic.py` computes an optical-flow-based "motion spike
score" — a genuine physical signature of many collisions (sudden, sharp
motion change) that needs **no training data** to work.

`controllers/controller.py` now blends this with the model's probability:

```python
final_score = (MODEL_ENSEMBLE_WEIGHT * model_probability) +
              ((1 - MODEL_ENSEMBLE_WEIGHT) * motion_spike_score)
```

Tune `MODEL_ENSEMBLE_WEIGHT` in `config.py`:
- Lower it (e.g. 0.3) while your model is undertrained — lean on the heuristic more.
- Raise it toward 1.0 as you train on more real, high-quality labeled footage.

The dashboard and CLI now clearly print **"ACCIDENT DETECTED"** or
**"NO ACCIDENT DETECTED"** based on the combined score, along with a
breakdown of each individual signal.

**Important honesty note:** neither the motion heuristic nor an undertrained
model is a substitute for training on real accident footage. The heuristic
improves real-world usefulness *today*; retraining on CADP/DAD/CCD (see
below) is still the real path to high accuracy.

### 2. Firebase authentication

The Streamlit app is now gated behind email/password login via Firebase
Authentication (chosen over Supabase because Supabase's free tier caps
you at 2 active projects per organization — Firebase's free Spark tier
does not have that cap, and also gives you a free real-time database
if you want to log prediction history later).

**Setup:**
1. Create a free project at [console.firebase.google.com](https://console.firebase.google.com)
2. In the left sidebar: **Build → Authentication → Get Started**, then
   enable the **Email/Password** sign-in provider
3. Go to **Project Settings** (gear icon) → **General** tab
4. Under "Your apps", click the **Web icon (`</>`)** to register a web
   app (no Firebase Hosting required — this just generates a config)
5. Copy the **`apiKey`** value from the shown `firebaseConfig` snippet
6. Locally: copy `.streamlit/secrets.toml.example` to
   `.streamlit/secrets.toml` and fill in your real `FIREBASE_API_KEY`.
   This file is git-ignored — never commit it.
7. On Streamlit Community Cloud: go to your app → **Manage app → Settings
   → Secrets**, and paste the same `FIREBASE_API_KEY` value there.

Firebase's Web API key is designed to be used client-side (it identifies
your project, it isn't a secret admin credential) — but it's still good
practice to keep it out of your public repo via secrets, which is what
this setup does.

By default, Firebase does **not** require email confirmation before
first login (unlike Supabase) — configurable in the Firebase console
under **Authentication → Settings → User actions** if you want to
require it.

To disable login entirely (e.g. for local testing), set in `config.py`:
```python
REQUIRE_LOGIN = False
```

### 3. Cloud Firestore (prediction history, no paid Storage required)

Each analyzed video's result is now saved to Firestore, scoped to the
logged-in user. A "Your Prediction History" panel on the dashboard
lets users browse past results, including a small thumbnail image of
the detected accident-likely moment.

**Important — why there's no full video storage:** as of February 3,
2026, Cloud Storage for Firebase requires the paid Blaze plan, even
for default buckets — Google removed the free tier for it. Firestore
itself remains free (Spark plan). So instead of uploading and storing
the full video file, each prediction record includes a small
compressed JPEG thumbnail (the peak-moment frame, base64-encoded) —
comfortably within Firestore's free tier and well under its 1 MiB
per-document limit (typically tens of KB). If you later upgrade to
the Blaze plan and want full video playback in history, you can
reintroduce Firebase Storage (or a free third-party option like
Cloudinary or Backblaze B2) and store its URL alongside these
Firestore records.

This uses the **Firebase Admin SDK** with a service account (different
from the Web API key used for login) — the standard, correct choice
since this Streamlit app runs entirely server-side.

**Setup:**
1. In your Firebase project: **Build → Firestore Database → Create
   database** (start in test mode — see note on security rules below)
2. **Project Settings** (gear icon) → **Service accounts** tab → **Generate
   new private key** → a JSON file downloads
3. Open that JSON file and copy its fields into a `[firebase_service_account]`
   table in `.streamlit/secrets.toml` — see the template in
   `.streamlit/secrets.toml.example` for the exact field names and format
4. On Streamlit Community Cloud: paste the same `[firebase_service_account]`
   table into **Manage app → Settings → Secrets**

**Security note:** "test mode" Firestore rules allow open read/write
for a limited time — fine while developing, but before sharing your
app publicly, tighten your Firestore security rules (in the Firebase
console) to restrict access to each user's own records.

This feature is optional — if `[firebase_service_account]` isn't
configured, the app still works for detection, just without history.
Set `ENABLE_DB_STORAGE = False` in `config.py` to hide the feature
entirely.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare your data

```
sample_data/raw/
├── accident/
└── non_accident/
```
Recommended real datasets: **CADP**, **DAD**, **CCD**. For a quick pipeline
smoke-test with synthetic data first:
```bash
python make_dummy_data.py
```

## 2. Preprocess

```bash
python data/preprocessing.py
```

## 3. Train

```bash
python train.py
```

## 4. Run

Dashboard:
```bash
streamlit run views/app.py
```

CLI:
```bash
python predict.py --video path/to/clip.mp4
```

## Configuration

All hyperparameters, the motion-heuristic weight, and `REQUIRE_LOGIN` live
in `config.py`.

## Next steps / extensions

- Swap in real training data (CADP/DAD/CCD) — this remains the single
  biggest lever for real accuracy.
- Tune `MODEL_ENSEMBLE_WEIGHT` against real footage once you have some to
  calibrate against.
- Add row-level Firestore security rules restricting each user to
  their own data before sharing the app publicly.
- Add YOLOv8 + tracking for object-level motion analysis as a third
  ensemble signal.
