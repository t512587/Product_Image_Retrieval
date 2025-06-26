import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

class ModelManager:
    """模型管理器，負責模型載入和特徵提取"""
    
    def __init__(self, model_name="ViT-B-16", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model, self.preprocess = load_from_name(
            model_name, device=self.device, download_root="./"
        )
        self.model.eval()
    
    def extract_features(self, image_tensor):
        """提取圖像特徵向量"""
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features
    
    def preprocess_image(self, image):
        """預處理圖像"""
        return self.preprocess(image).unsqueeze(0).to(self.device)