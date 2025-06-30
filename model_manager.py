import torch
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
        self.model.float()    
    def extract_features(self, image_tensor):
        """提取全圖特徵向量 (CLS token)"""
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def extract_patch_features(self, image_tensor, layer_idx=4):
        """提取 patch-level 特徵，回傳 shape (B, num_patches, dim)"""
        with torch.no_grad():
            x = self.model.visual.conv1(image_tensor)  # (B, 768, 14, 14)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, 196, 768)

            cls_token = self.model.visual.class_embedding.to(x.dtype)
            cls_token = cls_token + torch.zeros(x.shape[0], 1, x.shape[2], device=self.device)
            x = torch.cat([cls_token, x], dim=1)  # (B, 197, 768)

            x = x + self.model.visual.positional_embedding.to(x.dtype)
            x = self.model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # (197, B, 768)

            for i, block in enumerate(self.model.visual.transformer.resblocks):
                x = block(x)
                if i == layer_idx:
                    break

            x = x.permute(1, 0, 2)  # (B, 197, 768)
            x = self.model.visual.ln_post(x)

            patch_tokens = x[:, 1:, :]  # 去掉 CLS token
            patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)  # normalize
            return patch_tokens  # shape: (B, 196, 768)

    def preprocess_image(self, image):
        return self.preprocess(image).unsqueeze(0).to(self.device).float()