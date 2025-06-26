import os

class Config:
    """配置類別，統一管理所有設定"""
    
    # 資料夾路徑
    IMAGE_DIR = "query_img"
    DATABASE_DIR = "data_base_img"
    OUTPUT_DIR = "output_images"

    
    # 字體設定
    FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
    FONT_SIZE = 24
    
    # 模型設定
    MODEL_NAME = "ViT-B-16"
    DEVICE = "cuda"  # 或 "cpu"
    
    # 檢索設定
    TOP_K = 3
    
    # 標籤字典
    ID2LABEL = {
        1: "奇多家常", 2: "樂事瑞士", 3: "樂事九州", 4: "樂事湖鹽", 5: "奇多兩倍",
        6: "舒跑", 7: "特上紅茶", 8: "特上檸檬茶", 9: "波蜜", 10: "FIN好菌",
        11: "FIN", 12: "義美檸檬", 13: "義美草莓", 14: "茶王烏龍", 15: "冰釀烏龍",
        16: "冷萃綠茶", 17: "冷韻清茶", 18: "蘋果紅茶", 19: "茶裏王青心烏龍",
        20: "茶裏王白毫烏龍", 21: "茶裏王英式紅茶", 22: "茶裏王台式綠茶", 23: "原萃"
    }