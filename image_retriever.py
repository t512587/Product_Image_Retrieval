import numpy as np
import torch

class ImageRetriever:
    """圖像檢索器，負責相似度計算和檢索"""
    
    def __init__(self, model_manager, patch_mode=False):
        self.model_manager = model_manager
        self.patch_mode = patch_mode
    
    def retrieve_similar_images(self, query_image, feature_db, name_db, label_db, topk=1):
        query_tensor = self.model_manager.preprocess_image(query_image)

        if not self.patch_mode:
            query_feature = self.model_manager.extract_features(query_tensor)
            # 強制轉float32，feature_db也轉float32
            query_feature = query_feature.float()
            feature_db = feature_db.float()
            similarities = (query_feature.cpu() @ feature_db.T.cpu()).squeeze(0).numpy()
        else:
            query_patch = self.model_manager.extract_patch_features(query_tensor)
            feature_db = feature_db.to(self.model_manager.device).to(self.model_manager.model.dtype)
            sims = []
            for db_patch in feature_db:
                sim_matrix = torch.matmul(query_patch[0], db_patch.T)
                max_sim, _ = sim_matrix.max(dim=1)
                avg_max_sim = max_sim.mean()
                sims.append(avg_max_sim.item())
            similarities = np.array(sims)

        topk_indices = np.argsort(similarities)[-topk:][::-1]
        results = []
        for idx in topk_indices:
            results.append({
                'index': idx,
                'name': name_db[idx],
                'label': int(label_db[idx]),
                'similarity': similarities[idx]
            })
        return results
