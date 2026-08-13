import torch
from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image
from src.config import Config

class FashionEmbedder:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = getattr(Config, "MODEL_NAME", "patrickjohncyh/fashion-clip")
            
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading embedding model '{model_name}' on device: {self.device}...")
        self.fclip = FashionCLIP(model_name)

    def embed_image(self, image: Image.Image) -> list[float]:
        """Generates a normalized 512-dimensional vector embedding for an image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Temporary save for FashionCLIP internal loader
        temp_path = "temp_embed.jpg"
        image.save(temp_path)

        try:
            image_embeddings = self.fclip.encode_images([temp_path], batch_size=1)
            # Flatten and normalize vector
            features = torch.tensor(image_embeddings[0])
            norm = features.norm(p=2, dim=-1, keepdim=True)
            if norm > 0:
                features = features / norm
            return features.cpu().numpy().flatten().tolist()
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)