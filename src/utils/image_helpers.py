import io
import requests
import base64
from PIL import Image

def load_image_from_input(image_input: str | Image.Image) -> Image.Image:
    """Loads a PIL Image from a file path, URL, base64 string, or PIL Image object."""
    # If already a PIL Image
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
        
    # If URL
    if image_input.startswith(("http://", "https://")):
        response = requests.get(image_input, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
        
    # If Base64 String
    if image_input.startswith("data:image") or len(image_input) > 500:
        if "," in image_input:
            image_input = image_input.split(",")[1]
        image_data = base64.b64decode(image_input)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
        
    # Assume local file path
    return Image.open(image_input).convert("RGB")