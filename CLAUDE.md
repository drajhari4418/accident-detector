# Claude Project Notes

## Project

This repository is an automated traffic accident detector built with Python,
PyTorch, OpenCV, and Streamlit. It follows an MVC-style structure:

- `models/model.py`: ResNet18 + LSTM accident classifier.
- `controllers/controller.py`: training, evaluation, checkpointing, and inference.
- `data/preprocessing.py`: raw video to fixed-length `.npy` frame sequences.
- `data/dataset.py`: PyTorch dataset and data loader helpers.
- `views/app.py`: Streamlit dashboard.
- `train.py`: command-line training entry point.
- `predict.py`: command-line video inference entry point.
- `config.py`: paths, model settings, thresholds, and hyperparameters.

## Common Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare sample or raw data:

```bash
python make_dummy_data.py
python data/preprocessing.py
```

Train the model:

```bash
python train.py
```

Run inference on a video:

```bash
python predict.py --video path/to/clip.mp4
```

Run the dashboard:

```bash
streamlit run views/app.py
```

## Development Notes

- Keep hyperparameters and filesystem paths centralized in `config.py`.
- Route PyTorch training and inference work through `AccidentDetectorController`
  instead of importing torch directly in the Streamlit view.
- Runtime outputs such as checkpoints and processed data may be large; avoid
  committing generated datasets or model weights unless intentionally needed.
- If no checkpoint exists, `predict.py` falls back to untrained weights for demo
  behavior only.
