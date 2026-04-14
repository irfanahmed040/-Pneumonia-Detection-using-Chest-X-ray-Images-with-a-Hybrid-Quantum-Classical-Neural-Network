from pathlib import Path
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.inference import Predictor


app = FastAPI(title="Adaptive Hybrid Quantum-Classical CNN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor(checkpoint_path="checkpoints/best_model.pt")

frontend_dir = Path("frontend")
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": predictor.is_ready()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read image.") from exc

    try:
        pred, confidence, probs = predictor.predict(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": probs,
    }
