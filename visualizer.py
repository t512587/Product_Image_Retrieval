import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os

class Visualizer:
    """視覺化器，負責繪製邊界框和標籤"""
    
    def __init__(self, font_path, font_size=24):
        self.font_path = font_path
        self.font_size = font_size
        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except:
            print(f"[警告] 無法載入字體 {font_path}，使用預設字體")
            self.font = ImageFont.load_default()
    
    def draw_predictions(self, image, bboxes, predictions, id2label):
        """在圖像上繪製預測結果"""
        # 轉換為OpenCV格式
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        for bbox, pred in zip(bboxes, predictions):
            # 繪製邊界框
            cv2.rectangle(image_cv, 
                         (bbox['xmin'], bbox['ymin']), 
                         (bbox['xmax'], bbox['ymax']), 
                         (0, 0, 255), 2)
            
            # 準備標籤文字
            label_text = id2label.get(pred['label'], "未知")
            # 轉回PIL格式繪製中文文字
            image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            
            # 計算文字位置
            text_pos = (bbox['xmin'], max(bbox['ymin'] - 30, 0))
            draw.text(text_pos, label_text, font=self.font, fill=(255, 255, 0))
            
            # 轉回OpenCV格式
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        return image_cv
    
    def save_image(self, image_cv, save_path):
        """儲存圖像"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image_cv)