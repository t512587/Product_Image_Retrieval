from ultralytics import YOLO
from PIL import Image
import os

class YOLOParser:
    """使用 YOLOv8 模型進行即時物件偵測（含信心值，不含 class_id）"""

    def __init__(self, model_path=r"C:\Users\admin\Desktop\yolo\runs\detect\train3\weights\best.pt", device="cuda"):
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)

    def detect_objects(self, image_input, conf_threshold=0.25, filename=None):
        # 判斷 filename
        if filename is None:
            if hasattr(image_input, 'filename'):
                filename = os.path.basename(image_input.filename)
            elif isinstance(image_input, str):
                filename = os.path.basename(image_input)
            else:
                filename = "unknown"

        # 讀取圖片（如果是 PIL Image 就直接用）
        if isinstance(image_input, Image.Image):
            img = image_input
        else:
            img = Image.open(image_input).convert("RGB")

        img_width, img_height = img.size

        # 進行模型推論
        results = self.model(img)[0]  # 取第一張圖的結果

        objects = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue  # 濾掉低信心的框

            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

            obj_info = {
                'xmin': max(0, xmin),
                'ymin': max(0, ymin),
                'xmax': min(img_width - 1, xmax),
                'ymax': min(img_height - 1, ymax),
                'confidence': conf
            }
            objects.append(obj_info)

        return filename, objects
