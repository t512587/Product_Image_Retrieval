from pipeline import ImageRetrievalPipeline
from config import Config

# 初始化 pipeline
pipeline = ImageRetrievalPipeline(Config())
pipeline.build_database()  # 初始化一次即可

# 預測
results = pipeline.predict_objects_from_image(r"C:\Users\admin\Desktop\Product_Image_Retrieval\test_img\messageImage_1750818002553.jpg",save_vis=True)

for i, item in enumerate(results):
    print(f"物件{i+1}: 座標: {item['bbox']}，預測名稱: {item['name']}，信心值: {item['confidence']:.2f}")
