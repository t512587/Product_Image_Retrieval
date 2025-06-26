import os
import torch
from PIL import Image

class DatabaseBuilder:
    """資料庫建構器，負責建立特徵資料庫"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def build_feature_database(self, image_folder):
        """建立圖像特徵資料庫 - 支援數字子資料夾結構"""
        image_features_list = []
        image_names_list = []
        image_labels = []
        
        print(f"正在掃描資料庫資料夾: {image_folder}")
        
        # 遍歷主資料夾下的所有子資料夾
        for subfolder_name in os.listdir(image_folder):
            subfolder_path = os.path.join(image_folder, subfolder_name)
            
            # 確認是資料夾且名稱是數字
            if not os.path.isdir(subfolder_path):
                continue
                
            if not subfolder_name.isdigit():
                print(f"[跳過] 非數字資料夾: {subfolder_name}")
                continue
            
            label = int(subfolder_name)  # 資料夾名稱作為標籤
            print(f"正在處理標籤 {label} 的圖片...")
            
            # 處理該子資料夾下的所有圖片
            image_count = 0
            for fname in os.listdir(subfolder_path):
                if not self._is_image_file(fname):
                    continue
                    
                try:
                    image_path = os.path.join(subfolder_path, fname)
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = self.model_manager.preprocess_image(image)
                    
                    features = self.model_manager.extract_features(image_tensor)
                    
                    image_features_list.append(features.cpu())
                    image_names_list.append(f"{subfolder_name}/{fname}")  # 保存相對路徑
                    image_labels.append(label)
                    image_count += 1
                    
                except Exception as e:
                    print(f"[錯誤] 無法處理 {subfolder_name}/{fname}: {e}")
            
            print(f"✅ 標籤 {label}: 處理了 {image_count} 張圖片")
        
        if len(image_features_list) == 0:
            raise ValueError("資料庫圖片為空或格式錯誤，請檢查資料夾結構")
        
        image_features_all = torch.cat(image_features_list, dim=0)
        
        # 統計每個標籤的圖片數量
        label_counts = {}
        for label in image_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"✅ 成功建立資料庫！")
        print(f"總共處理 {len(image_names_list)} 張圖片")
        print(f"標籤分布: {dict(sorted(label_counts.items()))}")
        
        return image_features_all, image_names_list, image_labels
    
    def _is_image_file(self, filename):
        """檢查是否為圖片檔案"""
        return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))