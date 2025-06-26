import numpy as np
import torch

class ImageRetriever:
    """圖像檢索器，負責相似度計算和檢索"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def retrieve_similar_images(self, query_image, feature_db, name_db, label_db, topk=1):
        """檢索相似圖像"""
        # 預處理查詢圖像
        query_tensor = self.model_manager.preprocess_image(query_image)
        
        # 提取特徵
        query_feature = self.model_manager.extract_features(query_tensor)
        
        # 計算相似度
        similarities = (query_feature.cpu().float() @ feature_db.T.float()).squeeze(0).numpy()
        
        # 取得Top-K結果
        topk_indices = np.argsort(similarities)[-topk:][::-1]
        
        results = []
        for idx in topk_indices:
            result = {
                'index': idx,
                'name': name_db[idx],
                'label': int(label_db[idx]),
                'similarity': similarities[idx]
            }
            results.append(result)
        
        return results