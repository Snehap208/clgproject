import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Brief descriptions to give Gemini context about each disease
_DISEASE_CONTEXT = {
    "Melanoma":                "a potentially life-threatening skin cancer arising from melanocyte cells.",
    "Melanocytic Nevi":        "a common benign mole formed by melanocyte cells, usually harmless.",
    "Basal Cell Carcinoma":    "the most common skin cancer; slow-growing and rarely spreads beyond the skin.",
    "Actinic Keratosis":       "a pre-cancerous rough, scaly patch caused by years of sun exposure.",
    "Benign Keratosis":        "a non-cancerous skin growth such as seborrheic keratosis or solar lentigo.",
    "Dermatofibroma":          "a benign nodule of fibrous tissue, usually appearing on the legs.",
    "Vascular Lesion":         "an abnormality of blood vessels on or just under the skin surface.",
    "Squamous Cell Carcinoma": "the second most common skin cancer; can spread if left untreated.",
}


def get_explanation(predicted_class: str, confidence: float, top3: list) -> str:
    """
    Ask Gemini to generate a concise, doctor-style medical explanation
    for the given skin disease prediction.
    """
    disease_ctx = _DISEASE_CONTEXT.get(predicted_class, "a detected skin condition.")
    alternatives = ", ".join(
        f"{item['class_name']} ({item['confidence']:.1f}%)"
        for item in top3[1:]
    )

    prompt = f"""You are an expert dermatology AI assistant. A deep-learning model (ResNet50 trained on ISIC 2019) has analysed a dermoscopy image and returned:

Primary diagnosis : {predicted_class} — {disease_ctx}
Confidence        : {confidence:.1f}%
Differential      : {alternatives}

Write a concise, medically accurate explanation (4-6 sentences) in plain English that:
1. Briefly describes what {predicted_class} is.
2. Mentions common visual characteristics to look for.
3. States the urgency level (requires urgent care / routine check / monitor at home).
4. Reminds the user this is an AI tool and to consult a licensed dermatologist.

Do NOT use markdown, bullet points, or headers. Write as one flowing paragraph only."""

    try:
        response = _client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"AI explanation unavailable: {e}"
