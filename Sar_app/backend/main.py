"""
SAR Image Classifier — FastAPI Backend
Classifies SAR (Synthetic Aperture Radar) images into 8 MSTAR military vehicle classes
using an EfficientNet B0 model hosted on Hugging Face.
"""

import io
from pathlib import Path
from typing import Optional, List
import torch
import torch.nn as nn
import timm
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from openai import OpenAI
from pydantic import BaseModel

# ──────────────────────────── Paths ────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent  # Sar_app/
FRONTEND_DIR = BASE_DIR / "frontend"

# ──────────────────────────── Config ────────────────────────────

CLASSES = ["2S1", "BRDM_2", "BTR_60", "D7", "SLICY", "T62", "ZIL131", "ZSU_23_4"]

VEHICLE_INFO = {
    "2S1":      {"full_name": "2S1 Gvozdika",      "type": "Self-propelled Howitzer",         "origin": "Soviet Union"},
    "BRDM_2":   {"full_name": "BRDM-2",             "type": "Armored Scout Car",               "origin": "Soviet Union"},
    "BTR_60":   {"full_name": "BTR-60",              "type": "Armored Personnel Carrier",       "origin": "Soviet Union"},
    "D7":       {"full_name": "Caterpillar D7",      "type": "Military Bulldozer",              "origin": "United States"},
    "SLICY":    {"full_name": "SLICY",               "type": "Calibration / Test Target",       "origin": "United States"},
    "T62":      {"full_name": "T-62",                "type": "Main Battle Tank",                "origin": "Soviet Union"},
    "ZIL131":   {"full_name": "ZIL-131",             "type": "Military Cargo Truck",            "origin": "Soviet Union"},
    "ZSU_23_4": {"full_name": "ZSU-23-4 Shilka",    "type": "Self-propelled Anti-Aircraft Gun", "origin": "Soviet Union"},
}

# ──────────────────────────── OpenRouter Client ────────────────────────────

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

SAR_SYSTEM_PROMPT = """You are an expert AI assistant specialized in Synthetic Aperture Radar (SAR) imagery and military vehicle recognition. You are integrated into a SAR Target Classifier application that uses an EfficientNet B0 model trained on the MSTAR (Moving and Stationary Target Acquisition and Recognition) dataset.

Your knowledge includes:

**MSTAR Dataset:**
- Collected by Sandia National Laboratories using an X-band SAR sensor
- Contains SAR images of military vehicles at various depression angles (15° and 17° commonly used)
- Standard benchmark for SAR Automatic Target Recognition (ATR) research
- Images are 128×128 pixel chips of individual targets

**8 Target Classes in this classifier:**
1. **2S1 Gvozdika** — Soviet self-propelled 122mm howitzer, amphibious, used extensively in Cold War era
2. **BRDM-2** — Soviet amphibious armored scout car used for reconnaissance, crew of 4
3. **BTR-60** — Soviet 8×8 armored personnel carrier, can carry 16 troops
4. **Caterpillar D7** — US military bulldozer used for engineering and construction tasks
5. **SLICY** — Calibration/test target used to validate SAR imaging systems
6. **T-62** — Soviet main battle tank with 115mm smoothbore gun, successor to T-55
7. **ZIL-131** — Soviet 6×6 military cargo truck, 3.5 tonne payload capacity
8. **ZSU-23-4 Shilka** — Soviet self-propelled anti-aircraft gun with quad 23mm autocannons and radar

**SAR Technology:**
- SAR uses microwave radar to create high-resolution images regardless of weather or lighting
- Works by synthesizing a large antenna aperture through platform motion
- Advantages: all-weather, day/night capability, can penetrate clouds and foliage
- Applications: military reconnaissance, environmental monitoring, disaster assessment

**EfficientNet B0 Model:**
- Uses compound scaling to balance network depth, width, and resolution
- Fine-tuned with a custom classifier head: Linear(1280→256) → ReLU → Dropout(0.4) → Linear(256→8)
- Input preprocessing: resize to 224×224, convert to grayscale then 3-channel, ImageNet normalization

When the user has just classified an image, you will receive the prediction context. Use it to provide insightful analysis about the predicted vehicle, confidence scores, and what SAR features may distinguish the targets.

Keep responses concise, informative, and focused on SAR/MSTAR topics. Use markdown formatting for clarity when appropriate."""

# ──────────────────────────── Chat Models ────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    prediction_context: Optional[str] = None

# ──────────────────────────── Model Setup ────────────────────────────

print("⏳ Downloading model from Hugging Face Hub...")
model_path = hf_hub_download(
    repo_id="ramanjanm/efficientnet_mstar",
    filename="efficientnet_mstar.pt"
)
print(f"✅ Model downloaded to cache: {model_path}")

print("⏳ Loading EfficientNet B0 model...")
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 8)
)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()
print("✅ Model loaded and ready for inference!")

# ──────────────────────────── Preprocessing ────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ──────────────────────────── FastAPI App ────────────────────────────

app = FastAPI(
    title="SAR Target Classifier",
    description="Classify SAR images into 8 MSTAR military vehicle classes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded SAR image, run inference,
    and return the predicted class + confidence scores.
    """
    try:
        # Read and preprocess the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Build response
        confidences = {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(CLASSES, probabilities)
        }

        # Sort by confidence (descending)
        sorted_confidences = dict(
            sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        )

        predicted_class = max(confidences, key=confidences.get)

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidences[predicted_class],
            "all_confidences": sorted_confidences,
            "vehicle_info": VEHICLE_INFO[predicted_class],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that uses OpenRouter API to answer SAR-related questions.
    Accepts conversation history and optional prediction context.
    """
    try:
        # Build the system message with optional prediction context
        system_content = SAR_SYSTEM_PROMPT
        if request.prediction_context:
            system_content += f"\n\n**Current Prediction Context:**\n{request.prediction_context}"

        # Build messages for the API call
        api_messages = [{"role": "system", "content": system_content}]
        for msg in request.messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        # Call OpenRouter API
        response = openrouter_client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=api_messages,
        )

        assistant_message = response.choices[0].message.content

        return {
            "success": True,
            "response": assistant_message,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
