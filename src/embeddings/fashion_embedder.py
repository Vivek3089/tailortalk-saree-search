import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from src.config import Config

class FashionEmbedder:
    def __init__(self, model_name: str = Config.MODEL_NAME):
        # Hardware acceleration setup (Apple Silicon MPS / Nvidia CUDA / CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Loading embedding model '{model_name}' on device: {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_image(self, image: Image.Image) -> list[float]:
        """Converts a PIL Image into a normalized vector embedding list."""
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
            # Safely extract raw torch.Tensor regardless of return wrapper
            if isinstance(outputs, torch.Tensor):
                features = outputs
            elif hasattr(outputs, "image_embeds"):
                features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs[0]
            
            # Normalize vector for accurate cosine similarity matching
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features.cpu().numpy().flatten().tolist()