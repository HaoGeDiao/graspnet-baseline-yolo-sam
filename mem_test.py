import torch
from ultralytics import YOLO

def measure_memory(model_path):
    torch.cuda.empty_cache()
    # 记录初始显存
    mem_before = torch.cuda.memory_allocated()
    
    # 加载模型
    model = YOLO(model_path).cuda()
    
    # 记录峰值显存
    mem_after = torch.cuda.max_memory_allocated()
    print(f"{model_path}显存占用: {(mem_after - mem_before)/1024**2:.2f} MB")

# 测试SAM_b模型
measure_memory("models/yolo_and_sam/sam_b.pt") 

# 测试YOLOv8s-world模型
measure_memory("models/yolo_and_sam/yolov8s-world.pt")