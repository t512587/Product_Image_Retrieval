from config import Config
from pipeline import ImageRetrievalPipeline
def main():
    """主函數"""
    import os
    print("當前工作目錄：", os.getcwd())

    # 載入配置
    config = Config()
    
    # 建立管道
    pipeline = ImageRetrievalPipeline(config)
    
    # 執行管道
    pipeline.run()

if __name__ == "__main__":
    main()