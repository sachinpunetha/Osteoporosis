"""
FastAPI backend for knee X-ray osteoporosis classification.

Serves predictions from the final model: ResNet18, trained on leg-split-corrected
data with regularization (dropout + weight decay + reduced photometric jitter).

Run locally to test:
    uvicorn app:app --host 0.0.0.0 --port 8000

Then POST an image to http://localhost:8000/predict

IMPORTANT: this is a research/prototype tool. It is NOT a medical device and
should not be presented as providing clinical diagnoses.
"""

import io
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================== CONFIG ==============================
CHECKPOINT_DIR = "./checkpoints"   # put your 5 fold .pth files here (resnet18_fold0_singleleg_reg.pth ... fold4)
CLASSES = ["Normal", "Osteopenia", "Osteoporosis"]
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
NUM_FOLDS = 5
MODEL_SUFFIX = "_singleleg_reg_checkpoint"   # final model: leg-split-corrected data + regularization
HAS_DROPOUT_HEAD = True            # this run's head is Sequential(Dropout(0.4), Linear) -- must match training

# Border-cropping thresholds -- must match crop_images.py exactly, since the
# leg-split training data was built from border-cropped images first.
BORDER_INTENSITY_THRESHOLD = 15
BORDER_ROW_COL_FRACTION = 0.98
MARGIN_PX = 5

# Double-leg detection thresholds -- must match split_double_leg.py. The model
# was trained only on single-leg images, so a double-leg upload would be
# out-of-distribution; we detect and warn rather than silently mispredict.
BONE_BRIGHTNESS_THRESHOLD = 40
GAP_MIN_WIDTH_FRACTION = 0.03
CENTER_SEARCH_FRACTION = 0.5
LEG_SPLIT_MARGIN_PX = 5   # matches split_double_leg.py's MARGIN_PX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


# ============================== BORDER CROPPING (mirrors crop_images.py) ==============================
def find_content_bbox(gray_arr):
    h, w = gray_arr.shape
    is_background = gray_arr < BORDER_INTENSITY_THRESHOLD

    row_bg_fraction = is_background.mean(axis=1)
    col_bg_fraction = is_background.mean(axis=0)

    content_rows = np.where(row_bg_fraction < BORDER_ROW_COL_FRACTION)[0]
    content_cols = np.where(col_bg_fraction < BORDER_ROW_COL_FRACTION)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return 0, h, 0, w

    top, bottom = content_rows[0], content_rows[-1] + 1
    left, right = content_cols[0], content_cols[-1] + 1

    top = max(0, top - MARGIN_PX)
    left = max(0, left - MARGIN_PX)
    bottom = min(h, bottom + MARGIN_PX)
    right = min(w, right + MARGIN_PX)

    return top, bottom, left, right


def crop_border(pil_img):
    """Trim uniform dark borders, matching the training preprocessing exactly."""
    gray_arr = np.array(pil_img.convert("L"))
    top, bottom, left, right = find_content_bbox(gray_arr)

    orig_area = gray_arr.shape[0] * gray_arr.shape[1]
    crop_area = (bottom - top) * (right - left)
    if crop_area < 0.5 * orig_area:
        return pil_img  # fallback: crop looked wrong, skip it

    return pil_img.crop((left, top, right, bottom))


# ============================== DOUBLE-LEG DETECTION & SPLITTING (mirrors split_double_leg.py) ==============================
def detect_double_leg_split(gray_arr):
    """
    Returns the column index to split at if this looks like a double-leg image,
    else None. Mirrors the training-time detection exactly so inference-time
    handling matches what the model was effectively trained on (single-leg images).
    """
    h, w = gray_arr.shape
    col_brightness = gray_arr.mean(axis=0)

    search_start = int(w * (1 - CENTER_SEARCH_FRACTION) / 2)
    search_end = int(w * (1 + CENTER_SEARCH_FRACTION) / 2)
    center_slice = col_brightness[search_start:search_end]

    is_dark = center_slice < BONE_BRIGHTNESS_THRESHOLD
    if is_dark.sum() < GAP_MIN_WIDTH_FRACTION * w:
        return None

    dark_indices = np.where(is_dark)[0]
    if len(dark_indices) == 0:
        return None

    gap_center = search_start + int(np.median(dark_indices))
    left_region = col_brightness[:gap_center]
    right_region = col_brightness[gap_center:]

    left_has_bone = (left_region > BONE_BRIGHTNESS_THRESHOLD).sum() > 0.15 * len(left_region)
    right_has_bone = (right_region > BONE_BRIGHTNESS_THRESHOLD).sum() > 0.15 * len(right_region)

    if left_has_bone and right_has_bone:
        return gap_center
    return None


# ============================== LOAD MODELS ONCE AT STARTUP ==============================
def load_fold_model(fold_idx):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"resnet18_fold{fold_idx}{MODEL_SUFFIX}.pth")
    state_dict = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    model = models.resnet18(weights=None)
    if HAS_DROPOUT_HEAD:
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(model.fc.in_features, len(CLASSES)))
    else:
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


print("Loading ensemble models...")
fold_models = [load_fold_model(i) for i in range(NUM_FOLDS)]
print(f"Loaded {len(fold_models)} models on {DEVICE}.")

# ============================== FASTAPI APP ==============================
app = FastAPI(title="Knee X-ray Osteoporosis Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your specific frontend domain before going live
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict
    disclaimer: str = (
        "This is a research prototype, not a certified medical device. "
        "Do not use for clinical decision-making without professional review."
    )


def predict_probs(pil_img):
    """Run the 5-fold ensemble on a single (already cropped) PIL image, returning
    the averaged probability array (numpy, shape (len(CLASSES),))."""
    input_tensor = eval_transform(pil_img).unsqueeze(0).to(DEVICE)
    probs_sum = torch.zeros(len(CLASSES))
    with torch.no_grad():
        for model in fold_models:
            probs_sum += model(input_tensor).softmax(dim=1).cpu()[0]
    return (probs_sum / NUM_FOLDS).numpy()


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Knee X-ray classifier API is running.", "model": MODEL_SUFFIX}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    img = crop_border(img)  # match training preprocessing

    # Check for double-leg framing. If detected, split into two single-leg
    # images (same approach used to build the training data), predict each
    # half independently, and average the results. This is handled entirely
    # server-side -- the API response looks identical either way.
    gray_arr = np.array(img.convert("L"))
    split_col = detect_double_leg_split(gray_arr)

    if split_col is not None:
        w = gray_arr.shape[1]
        left_crop = img.crop((0, 0, min(split_col + LEG_SPLIT_MARGIN_PX, w), gray_arr.shape[0]))
        right_crop = img.crop((max(split_col - LEG_SPLIT_MARGIN_PX, 0), 0, w, gray_arr.shape[0]))

        probs_left = predict_probs(left_crop)
        probs_right = predict_probs(right_crop)
        probs_avg = (probs_left + probs_right) / 2

        print(f"[predict] Double-leg image detected and split internally "
              f"(left pred: {CLASSES[probs_left.argmax()]}, right pred: {CLASSES[probs_right.argmax()]})")
    else:
        probs_avg = predict_probs(img)

    pred_idx = probs_avg.argmax()

    return PredictionResponse(
        predicted_class=CLASSES[pred_idx],
        confidence=float(probs_avg[pred_idx]),
        probabilities={cls: float(p) for cls, p in zip(CLASSES, probs_avg)},
    )
