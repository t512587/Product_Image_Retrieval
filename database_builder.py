import os
from PIL import Image
import torch

class DatabaseBuilder:
    """資料庫建構器，負責建立特徵資料庫"""
    
    def __init__(self, model_manager, patch_mode=False):
        self.model_manager = model_manager
        self.patch_mode = patch_mode
    
    def build_feature_database(self, image_folder):
        """建立圖像特徵資料庫 - 支援數字子資料夾結構"""
        feature_list = []
        name_list = []
        label_list = []
        
        print(f"正在掃描資料庫資料夾: {image_folder}")
        
        for subfolder_name in os.listdir(image_folder):
            subfolder_path = os.path.join(image_folder, subfolder_name)
            if not os.path.isdir(subfolder_path):
                continue
            if not subfolder_name.isdigit():
                print(f"[跳過] 非數字資料夾: {subfolder_name}")
                continue
            
            label = int(subfolder_name)
            print(f"正在處理標籤 {label} 的圖片...")
            
            for fname in os.listdir(subfolder_path):
                if not self._is_image_file(fname):
                    continue
                try:
                    image_path = os.path.join(subfolder_path, fname)
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = self.model_manager.preprocess_image(image)
                    
                    if self.patch_mode:
                        patch_features = self.model_manager.extract_patch_features(image_tensor)  # (1, N, D)
                        feature_list.append(patch_features.squeeze(0).cpu())
                    else:
                        features = self.model_manager.extract_features(image_tensor)  # (1, D)
                        feature_list.append(features.cpu())
                    
                    name_list.append(f"{subfolder_name}/{fname}")
                    label_list.append(label)
                    
                except Exception as e:
                    print(f"[錯誤] 無法處理 {subfolder_name}/{fname}: {e}")
            
            print(f"✅ 標籤 {label}: 處理完成")
        
        if len(feature_list) == 0:
            raise ValueError("資料庫圖片為空或格式錯誤，請檢查資料夾結構")
        
        if self.patch_mode:
            # feature_list: List of (num_patches, dim) tensors
            # 合成 (num_images, num_patches, dim)
            feature_db = torch.stack(feature_list, dim=0)
        else:
            # feature_list: List of (1, dim)
            feature_db = torch.cat(feature_list, dim=0)
        
        print(f"✅ 成功建立資料庫，總圖片數: {len(name_list)}")
        return feature_db, name_list, label_list
    
    def _is_image_file(self, filename):
        return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))