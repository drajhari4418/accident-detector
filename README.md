# Automated Accident Detector (Deep Learning, CNN+LSTM)

A traffic accident detection system built with an **MVC architecture**:

```
accident_detector/
├── config.py                  # all paths & hyperparameters
├── data/
│   ├── preprocessing.py       # video -> frame sequence (.npy) extraction
│   └── dataset.py             # PyTorch Dataset / DataLoader
├── models/
│   └── model.py                # MODEL: ResNet18 CNN + LSTM classifier
├── controllers/
│   └── controller.py          # CONTROLLER: training/eval/inference logic
├── views/
│   └── app.py                  # VIEW: Streamlit dashboard
├── train.py                    # CLI: train the model
├── predict.py                   # CLI: run inference on one video
└── checkpoints/                # saved model weights (created at runtime)
```

## How the pieces fit together

- **Model** (`models/model.py`): `AccidentDetectorNet` — a pretrained ResNet18
  extracts per-frame features, an LSTM aggregates them over a 16-frame clip,
  and a small FC head outputs accident probability.
- **Controller** (`controllers/controller.py`): `AccidentDetectorController`
  is the only class that touches PyTorch directly. It owns training loops,
  checkpointing, and `predict_video()` / `predict_frame_window()` for inference.
- **View** (`views/app.py`): a Streamlit app that uploads a video, calls
  `controller.predict_video()`, and renders the result. It never imports
  `torch` directly — everything goes through the Controller.
- **Data layer** (`data/`): `preprocessing.py` turns raw videos into fixed-length
  frame sequences saved as `.npy`; `dataset.py` wraps those in a PyTorch
  `Dataset`/`DataLoader`.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare your data

Organize raw videos like this:

```
sample_data/raw/
├── accident/
│   ├── clip001.mp4
│   └── ...
└── non_accident/
    ├── clip001.mp4
    └── ...
```

Recommended public datasets to seed this with: **CADP**, **DAD** (Dashcam
Accident Dataset), **CCD** (Car Crash Dataset) — or your own labeled dashcam
footage.

Then build the processed index:

```bash
python data/preprocessing.py
```

This extracts a 16-frame sequence per video and writes
`sample_data/processed/index.csv` (path,label pairs).

## 2. Train

```bash
python train.py
```

Trains a ResNet18(frozen)+LSTM classifier, evaluates on a held-out split
each epoch, and saves the best checkpoint to `checkpoints/best_model.pt`.

## 3. Run inference

CLI:
```bash
python predict.py --video path/to/clip.mp4
```

Dashboard:
```bash
streamlit run views/app.py
```

## Configuration

All hyperparameters (sequence length, frame size, learning rate, alert
threshold, etc.) live in `config.py` — edit there rather than in code.

## Next steps / extensions

- Swap the CNN backbone for EfficientNet, or the whole pipeline for a 3D CNN
  (I3D/C3D) for end-to-end spatiotemporal learning.
- Add YOLOv8 + ByteTrack for object-level motion analysis as a complementary
  signal (sudden velocity change, trajectory overlap).
- Add `ALERT_CONSECUTIVE_WINDOWS` debouncing logic in the Controller for
  live camera streams to reduce false positives.
- Unfreeze the CNN backbone (`freeze_backbone=False`) and fine-tune end-to-end
  once you have more labeled data.


--Good news: your last command fixed that —

(venv) C:\Users\Admin\Downloads\accident_detector\accident_detector>

Now you're inside the venv. Just re-run the same commands now:

python make_dummy_data.py
python data/preprocessing.py
python train.py

They should work now since (venv) is active in your prompt.


streamlit run views/app.py

(at last to run application on browser).

One thing to remember going forward: every time you open a new terminal/CMD window to work on this project, you need to activate the venv first:

cd C:\Users\Admin\Downloads\accident_detector\accident_detector
venv\Scripts\activate