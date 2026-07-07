# Knee X-Ray Osteoporosis Classifier — API

FastAPI backend serving predictions from a ResNet18 model (5-fold ensemble),
trained to classify knee X-rays into **Normal**, **Osteopenia**, or **Osteoporosis**.

> ⚠️ **This is a research prototype, not a certified medical device.**
> Do not use for clinical decision-making without professional review.

## What's in this repo

```
.
├── app.py              # FastAPI backend
├── requirements.txt    # Python dependencies
├── checkpoints/        # Put your 5 trained model files here (see below)
└── README.md
```

## 1. Setup

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add your model weights

This app expects **5 fold checkpoint files** in the `checkpoints/` folder, named:

```
checkpoints/resnet18_fold0_singleleg_reg_checkpoint.pth
checkpoints/resnet18_fold1_singleleg_reg_checkpoint.pth
checkpoints/resnet18_fold2_singleleg_reg_checkpoint.pth
checkpoints/resnet18_fold3_singleleg_reg_checkpoint.pth
checkpoints/resnet18_fold4_singleleg_reg_checkpoint.pth
```

These come from the training notebook's checkpoint-export cell (the "final,
regularized, leg-split-corrected" model). Copy them from your Kaggle
notebook's `models/final/` output folder into `checkpoints/` here.

If you're using a different trained run (e.g. the plain cropped model instead
of the final regularized one), update `MODEL_SUFFIX` and `HAS_DROPOUT_HEAD`
at the top of `app.py` to match — see the comments in that file for details.

## 3. Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

On startup you should see:
```
Loading ensemble models...
Loaded 5 models on cpu.
```
(or `cuda` if you have a GPU available).

## 4. Test it

**Health check:**
```bash
curl http://localhost:8000/
```

**Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/your/xray.jpg"
```

Or open `http://localhost:8000/docs` for the interactive Swagger UI, where you
can upload an image directly through the browser.

## What the API does

1. **Border cropping** — trims uniform black padding from the uploaded image,
   matching the preprocessing used during training.
2. **Double-leg detection** — if the uploaded image shows two legs side by
   side (the model was trained only on single-leg images), it's automatically
   split into two single-leg images internally. Both halves are classified
   separately and their predictions are averaged into one final result. This
   is handled entirely server-side — the API response looks the same either
   way, no extra fields or flags.
3. **Ensemble prediction** — the image (or each half) is run through all 5
   fold models, and their softmax outputs are averaged for the final
   prediction.

## Response format

```json
{
  "predicted_class": "Osteopenia",
  "confidence": 0.71,
  "probabilities": {
    "Normal": 0.12,
    "Osteopenia": 0.71,
    "Osteoporosis": 0.17
  },
  "disclaimer": "This is a research prototype, not a certified medical device. ..."
}
```

## Deploying

For a public-facing deployment, consider:
- **Render** or **Railway** — push this repo, they build and host it
- **Hugging Face Spaces** — good for ML demos, supports FastAPI directly
- **Fly.io** — more control, via a Dockerfile

Before going live, tighten the CORS setting in `app.py`
(`allow_origins=["*"]`) to your actual frontend domain.
