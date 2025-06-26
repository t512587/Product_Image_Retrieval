import os
from PIL import Image
from model_manager import ModelManager
from database_builder import DatabaseBuilder
from yolo_parser import YOLOParser
from image_retriever import ImageRetriever
from visualizer import Visualizer
from bg_remover import remove_bg_return_pil


def find_image_file(image_dir, base_filename):
    """根據 base name 自動尋找圖片檔案（支援多種副檔名）"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        path = os.path.join(image_dir, base_filename + ext)
        if os.path.exists(path):
            return path
    return None


class ImageRetrievalPipeline:
    """整合所有組件 (YOLO版本)"""

    def __init__(self, config):
        self.config = config

        # 初始化各組件
        self.model_manager = ModelManager(config.MODEL_NAME, config.DEVICE)
        self.database_builder = DatabaseBuilder(self.model_manager)
        self.yolo_parser = YOLOParser()
        self.retriever = ImageRetriever(self.model_manager)
        self.visualizer = Visualizer(config.FONT_PATH, config.FONT_SIZE)

        self.feature_db, self.name_db, self.label_db = None, None, None

    def build_database(self):
        """建立特徵資料庫"""
        print("📦 正在建立資料庫特徵...")
        self.feature_db, self.name_db, self.label_db = \
            self.database_builder.build_feature_database(self.config.DATABASE_DIR)

    def process_single_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"[警告] 找不到圖片 {image_path}")
            return

        filename = os.path.basename(image_path)

        try:
            processed_img = remove_bg_return_pil(image_path)
            print("✅ 圖片去背完成")
        except Exception as e:
            print(f"[錯誤] 去背失敗，改用原圖: {e}")
            processed_img = Image.open(image_path).convert('RGB')

        # 用去背圖片做偵測
        filename, objects = self.yolo_parser.detect_objects(processed_img, conf_threshold=0.6, filename=filename)

        if not objects:
            print(f"[提示] 未在圖片 {filename} 偵測到任何物件")
            return

        # 讀取原始圖片，視覺化用
        original_img = Image.open(image_path).convert('RGB')

        predictions = []
        for i, obj in enumerate(objects):
            # 裁切時用去背圖片
            cropped = processed_img.crop((obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']))

            results = self.retriever.retrieve_similar_images(
                cropped, self.feature_db, self.name_db, self.label_db, self.config.TOP_K
            )

            best_result = results[0]
            predictions.append(best_result)

            print(f"[圖片: {filename}] 物件 {i} 的 Top-{self.config.TOP_K} 預測：")
            for rank, res in enumerate(results):
                pred_name = self.config.ID2LABEL.get(res['label'], "未知")
                print(f"    🟢 Top{rank+1}: {res['name']} | 相似度: {res['similarity']:.4f} | 類別: {pred_name}")

        # 用原始圖片做視覺化（畫框和標籤）
        result_image = self.visualizer.draw_predictions(original_img, objects, predictions, self.config.ID2LABEL)
        save_path = os.path.join(self.config.OUTPUT_DIR, filename)
        self.visualizer.save_image(result_image, save_path)
        print(f"✅ 已儲存：{save_path}")





    def process_all_images(self):
        """處理所有圖像（使用 YOLOv8 偵測）"""
        print("🔍 開始進行圖對圖檢索與標籤預測...")

        image_files = [f for f in os.listdir(self.config.IMAGE_DIR)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        for image_name in image_files:
            image_path = os.path.join(self.config.IMAGE_DIR, image_name)
            self.process_single_image(image_path)


    def run(self):
        """執行"""
        try:
            self.build_database()
            self.process_all_images()
            print("\n🎉 執行完成！")
        except Exception as e:
            print(f"❌ 執行失敗: {e}")
            raise
    def predict_objects_from_image(self, image_input, remove_bg=True, conf_threshold=0.6, save_vis=False):
        """
        提供給外部使用的預測函式，輸入圖片，回傳每個物件的座標與預測名稱。
        
        Args:
            image_input: 圖像路徑 或 PIL.Image
            remove_bg: 是否進行去背處理
            conf_threshold: YOLOv8 的信心閾值
            save_vis: 是否儲存視覺化結果圖

        Returns:
            List of dict: 每個物件包含 'bbox', 'name', 'confidence'
        """
        # --- 處理輸入與原始圖像 ---
        if isinstance(image_input, str):
            image_path = image_input
            filename = os.path.basename(image_path)
            try:
                processed_img = remove_bg_return_pil(image_path) if remove_bg else Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"[錯誤] 去背失敗，改用原圖: {e}")
                processed_img = Image.open(image_path).convert("RGB")
            original_img = Image.open(image_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            filename = os.path.basename(getattr(image_input, 'filename', "unknown.jpg"))
            try:
                processed_img = remove_bg_return_pil(image_input.filename) if remove_bg and hasattr(image_input, 'filename') else image_input.convert("RGB")
            except Exception as e:
                print(f"[錯誤] 去背失敗，改用原圖: {e}")
                processed_img = image_input.convert("RGB")
            original_img = image_input.convert("RGB")
        else:
            raise ValueError("image_input 必須是檔案路徑或 PIL.Image")

        # --- 偵測物件 ---
        _, objects = self.yolo_parser.detect_objects(processed_img, conf_threshold=conf_threshold, filename=filename)

        if not objects:
            return []

        predictions = []
        for obj in objects:
            cropped = processed_img.crop((obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']))
            results = self.retriever.retrieve_similar_images(
                cropped, self.feature_db, self.name_db, self.label_db, topk=self.config.TOP_K
            )
            best_result = results[0]
            pred_name = self.config.ID2LABEL.get(best_result['label'], "未知")

            # 保存預測資訊
            predictions.append({
                'bbox': {
                    'xmin': obj['xmin'],
                    'ymin': obj['ymin'],
                    'xmax': obj['xmax'],
                    'ymax': obj['ymax']
                },
                'name': pred_name,
                'label': best_result['label'], 
                'confidence': obj.get('confidence', None)
            })

        # --- 視覺化與儲存 ---
        if save_vis:
            result_image = self.visualizer.draw_predictions(original_img, objects, predictions, self.config.ID2LABEL)
            save_path = os.path.join(self.config.OUTPUT_DIR, filename)
            self.visualizer.save_image(result_image, save_path)
            print(f"✅ 已儲存視覺化圖像：{save_path}")

        return predictions


