import os
import time
import mimetypes
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def upload_image_to_storage(file_name: str, image_bytes: bytes) -> str:
    """Uploads image to Supabase Storage and returns the public URL."""
    # Ensure unique filename
    ext = os.path.splitext(file_name)[1] or ".jpg"
    unique_name = f"{int(time.time())}_{file_name}"
    
    # Detect mime type
    mime_type, _ = mimetypes.guess_type(file_name)
    if not mime_type:
        mime_type = "image/jpeg"

    # Upload to 'skin-images' bucket
    supabase.storage.from_("skin-images").upload(
        path=unique_name,
        file=image_bytes,
        file_options={"content-type": mime_type}
    )
    
    # Get public URL
    return supabase.storage.from_("skin-images").get_public_url(unique_name)

def save_upload(file_name: str, file_size: int, image_url: str) -> str:
    res = supabase.table("uploads").insert({
        "file_name":  file_name,
        "file_path":  image_url,
        "file_size":  file_size,
        "image_type": "skin_lesion"
    }).execute()
    return res.data[0]["id"]

def save_prediction(upload_id: str, result: dict, processing_ms: int):
    supabase.table("predictions").insert({
        "upload_id":        upload_id,
        "model_name":       "resnet50_isic2019",
        "prediction_label": result["predicted_class"],
        "confidence_score": result["confidence"] / 100.0,
        "probabilities":    result["all_probs"],
    }).execute()

def get_recent_predictions(limit: int = 10):
    res = supabase.table("predictions") \
        .select("*, uploads(file_name, file_path)") \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return res.data
