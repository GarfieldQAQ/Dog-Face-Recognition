import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def process_images(
    input_dir: str,
    detected_dir: str,
    undetected_dir: str,
    model_path: str,
    conf_threshold: float = 0.5
):
    """
    YOLO目标检测处理程序
    
    参数:
        input_dir (str): 输入图像根目录
        detected_dir (str): 检测到目标的输出目录
        undetected_dir (str): 未检测到目标的输出目录
        model_path (str): YOLO模型路径(.pt)
        conf_threshold (float): 置信度阈值(默认0.5)
    """
    # 初始化模型
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 获取支持的图像格式
    img_formats = {"jpg", "jpeg", "png", "bmp", "webp"}
    
    # 递归遍历输入目录
    input_path = Path(input_dir)
    all_images = list(input_path.rglob("*"))
    
    # 使用进度条
    pbar = tqdm(all_images, desc="Processing Images", unit="img")
    
    for img_path in pbar:
        if not img_path.is_file():
            continue
        
        # 检查文件格式
        if img_path.suffix[1:].lower() not in img_formats:
            continue
        
        # 计算相对路径
        relative_path = img_path.relative_to(input_path)
        
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 执行预测
        results = model.predict(img, conf=conf_threshold, verbose=False)
        
        # 获取检测结果
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # 创建输出目录
        detected_subdir = Path(detected_dir) / relative_path.parent
        undetected_subdir = Path(undetected_dir) / relative_path.parent
        detected_subdir.mkdir(parents=True, exist_ok=True)
        undetected_subdir.mkdir(parents=True, exist_ok=True)
        
        # 处理检测结果
        if len(boxes) > 0:
            # 保存所有检测框
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = img[y1:y2, x1:x2]
                
                # 生成唯一文件名
                save_name = f"{img_path.stem}_box{i}{img_path.suffix}"
                save_path = detected_subdir / save_name
                cv2.imwrite(str(save_path), cropped)
        else:
            # 保存未检测图像
            save_path = undetected_subdir / img_path.name
            cv2.imwrite(str(save_path), img)
        
        # 更新进度条描述
        pbar.set_postfix({
            "detected": len(boxes),
            "current": img_path.name
        })

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO目标检测与图像分类保存")
    parser.add_argument("--input", required=True, help="输入图像根目录路径")
    parser.add_argument("--detected", required=True, help="检测到目标的输出目录")
    parser.add_argument("--undetected", required=True, help="未检测到目标的输出目录")
    parser.add_argument("--model", required=True, help="YOLO模型路径(.pt)")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值(0-1)")
    
    args = parser.parse_args()
    
    print(f"硬件加速: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    process_images(
        input_dir=args.input,
        detected_dir=args.detected,
        undetected_dir=args.undetected,
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    print("\n处理完成！")
    print(f"检测结果保存在: {args.detected}")
    print(f"未检测图像保存在: {args.undetected}")