import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.config import Config

class FashionEmbedder:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = getattr(Config, "MODEL_NAME", "patrickjohncyh/fashion-clip")
            
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading Fashion-CLIP from HuggingFace ({model_name}) on device: {self.device}...")
        
        # Load directly via Transformers (No datasets / pyarrow dependency)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_image(self, image: Image.Image) -> list[float]:
        """Generates a normalized 512-dimensional vector embedding for an image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image input
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
            # L2 Normalize the embedding
            norm = image_features.norm(p=2, dim=-1, keepdim=True)
            if norm > 0:
                image_features = image_features / norm
                
            return image_features.cpu().numpy().flatten().tolist()