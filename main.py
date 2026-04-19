import time
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from model import predict
from database import upload_image_to_storage, save_upload, save_prediction, get_recent_predictions, supabase
from gemini import get_explanation

app = FastAPI(title="Skin Disease Classifier API")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def root():
    return {"status": "running", "message": "Skin Disease Classifier API"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_bytes = await file.read()
    file_size   = len(image_bytes)

    # Run ResNet50 model
    start  = time.time()
    result = predict(image_bytes)
    ms     = int((time.time() - start) * 1000)

    # Get Gemini AI explanation (non-blocking — graceful on failure)
    explanation = get_explanation(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        top3=result["top3"]
    )

    # Save to Supabase Storage & Database
    try:
        image_url = upload_image_to_storage(file.filename, image_bytes)
        upload_id = save_upload(file.filename, file_size, image_url)
        save_prediction(upload_id, result, ms)
    except Exception as e:
        print(f"DB error (non-fatal): {e}")

    return {
        "success":          True,
        "predicted_class":  result["predicted_class"],
        "predicted_code":   result["predicted_code"],
        "confidence":       result["confidence"],
        "top3":             result["top3"],
        "all_probs":        result["all_probs"],
        "processing_ms":    ms,
        "ai_explanation":   explanation
    }

@app.get("/history")
def history():
    try:
        data = get_recent_predictions(20)
        return {"success": True, "data": data}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
