from transformers import pipeline
from PIL import Image

# 全域只建立一次 pipe
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

def remove_bg_return_pil(input_path):
    img = Image.open(input_path).convert("RGBA")
    result = pipe(img)  # 執行去背

    if isinstance(result, Image.Image):
        bg = Image.new("RGBA", result.size, (255, 255, 255, 255))
        combined = Image.alpha_composite(bg, result).convert("RGB")
        return combined
    else:
        raise ValueError(f"[錯誤] 預期是去背後的 PIL 圖像，但得到: {type(result)}")
