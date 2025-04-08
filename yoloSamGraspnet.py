import os
import sys
import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from graspnetAPI import GraspGroup
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# ==================== 硬编码配置参数 ====================
# 处理参数
NUM_POINT = 20000      # 点云采样数量
NUM_VIEW = 300        # 视角数量
COLLISION_THRESH = 0.01 # 碰撞检测阈值
VOXEL_SIZE = 0.01      # 体素大小
#数据和模型路径
DATA_DIR = 'cv_try'  # 数据目录，应该包含color.png和depth.png
GRASP_CHECKPOINT_PATH = 'pretrain/checkpoint-rs.tar'  # grasp权重
YOLO_MODEL = 'models/yolo_and_sam/yolov8s-world.pt'
SAM_MODEL = 'models/yolo_and_sam/mobile_sam.pt'

# 相机内参（使用深度相机参数生成点云）
DEPTH_INTR = {
    "ppx": 319.304,  # cx
    "ppy": 236.915,  # cy
    "fx": 387.897,  # fx
    "fy": 387.897  # fy
}
DEPTH_FACTOR = 1000.0  # 深度因子，根据实际数据调整 1

# ==================== mask后处理 ====================
def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask

#取前n的连通面积
def retain_topn_area(mask,n):
    # 连通域分析
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if retval <= 1:  # 无有效连通域
        return mask
    
    # 提取面积并排序
    areas = stats[1:, 4]
    sorted_indices = np.argsort(areas)[::-1]
    sorted_areas = areas[sorted_indices]
    
    # 计算累加阈值
    total = np.sum(sorted_areas)
    cumulative = np.cumsum(sorted_areas) / total
    cutoff = np.argmax(cumulative >= n)
    
    # 生成新掩膜
    new_mask = np.zeros_like(labels)
    valid_labels = sorted_indices[:cutoff+1] + 1
    for lbl in valid_labels:
        new_mask[labels == lbl] = 255
        
    return new_mask
# ==================== 分割模型选择 ====================
def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = SAM_MODEL
    overrides = dict(
        task='segment',
        mode='predict',
        imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    return SAMPredictor(overrides=overrides)

def set_classes(model, target_class):
    """Set YOLO-World model to detect specific class"""
    model.set_classes([target_class])

# ==================== yoloworld 分割 ====================
def detect_objects(image_path, target_class=None):
    """
    Detect objects with YOLO-World
    Returns: (list of bboxes in xyxy format, detected classes list, visualization image)
    """
    model = YOLO(YOLO_MODEL)
    if target_class:
        set_classes(model, target_class)

    results = model.predict(image_path)
    boxes = results[0].boxes
    vis_img = results[0].plot()  # Get visualized detection results

    # Extract valid detections
    valid_boxes = []
    for box in boxes:
        if box.conf.item() > 0.25:  # Confidence threshold
            valid_boxes.append({
                "xyxy": box.xyxy[0].tolist(),
                "conf": box.conf.item(),
                "cls": results[0].names[box.cls.item()]
            })

    return valid_boxes, vis_img

# ==================== sam 分割 ====================
def segment_image(image_path, output_mask=DATA_DIR+'/mask1.png'):
    # User input for target class
    use_target_class = input("Detect specific class? (yes/no): ").lower() == 'yes'
    target_class = input("Enter class name: ").strip() if use_target_class else None

    # Detect objects
    detections, vis_img = detect_objects(image_path, target_class)
    cv2.imwrite(DATA_DIR+'detection_visualization.jpg', vis_img)

    # Prepare SAM predictor
    predictor = choose_model()
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)  # Set image for SAM

    if detections:
        # Auto-select highest confidence bbox
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
    else:
        # Manual point selection
        print("No detections - click on target object")
        cv2.imshow('Select Object', vis_img)
        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                cv2.destroyAllWindows()

        cv2.setMouseCallback('Select Object', click_handler)
        cv2.waitKey(0)

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    # Save results
    if mask is not None:
        mask = retain_topn_area(mask,0.8)
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"Segmentation saved to {output_mask}")
    else:
        print("mask1")

    return mask
# ==================== 网络定义 ====================
def get_net():
    net = GraspNet(
        input_feature_dim=0,
        num_view=NUM_VIEW,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    checkpoint = torch.load(GRASP_CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


# ==================== 数据处理 ====================
def get_and_process_data(data_dir):
    # 加载原始数据
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0

    # 加载深度图并打印信息
    depth_img = Image.open(os.path.join(data_dir, 'depth.png'))
    depth = np.array(depth_img)

    # print("\n=== 深度图分析 ===")
    # print("图像格式:", depth_img.format)
    # print("存储模式:", depth_img.mode)
    # print("NumPy数组形状:", depth.shape)
    # print("数据类型:", depth.dtype)
    # print("最小值:", np.min(depth))
    # print("最大值:", np.max(depth))
    # print("非零像素数量:", np.count_nonzero(depth))
    # print("零值像素占比: %.2f%%" % (100 * (1 - np.count_nonzero(depth) / depth.size)))

    # # 深度因子分析建议
    # max_depth = np.max(depth)
    # suggested_factors = []
    # if max_depth > 10:  # 如果最大值较大，可能以毫米为单位存储
    #     suggested_factors.append(1000)  # 毫米转米
    # if max_depth < 10:  # 如果值很小，可能是以米为单位存储的浮点数
    #     suggested_factors.append(1.0)

    # print("\n深度因子建议：")
    # print(f"当前使用的深度因子: {DEPTH_FACTOR}")
    # if suggested_factors:
    #     print("检测到可能需要的深度因子：")
    #     for f in suggested_factors:
    #         print(f"-> {f} （解释：{'毫米转米' if f == 1000 else '直接使用米'}）")
    # else:
    #     print("无法自动推断深度因子，请手动验证")

    # 其余处理保持不变...
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'mask1.png')))

    # 验证图像尺寸
    print("\n=== 尺寸验证 ===")
    print("深度图尺寸:", depth.shape[::-1])  # (width, height)
    print("颜色图尺寸:", color.shape[:2][::-1])
    print("相机参数预设尺寸:", (1280, 720))

    # 创建相机参数对象
    camera = CameraInfo(
        width=1280,
        height=720,
        fx=DEPTH_INTR['fx'],
        fy=DEPTH_INTR['fy'],
        cx=DEPTH_INTR['ppx'],
        cy=DEPTH_INTR['ppy'],
        scale=DEPTH_FACTOR
    )

    # 生成点云
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # 应用掩码
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # 点云采样
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]

    # 转换为Open3D点云（用于可视化）
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # 转换为Tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)

    end_points = {'point_clouds': cloud_sampled}
    return end_points, cloud_o3d


# ==================== 碰撞检测 ====================
def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]


# ==================== 抓取位姿打印 ====================
def print_grasp_poses(gg):
    print(f"\nTotal grasps after collision detection: {len(gg)}")
    for i, grasp in enumerate(gg):
        print(f"\nGrasp {i + 1}:")
        print(f"Position (x,y,z): {grasp.translation}")
        print(f"Rotation Matrix:\n{grasp.rotation_matrix}")
        print(f"Score: {grasp.score:.4f}")
        print(f"Width: {grasp.width:.4f}")


# ==================== 主流程 ====================
def demo(data_dir):
    #yoloworld+mobile_sam分割
    segment_image(DATA_DIR+'/color.png')

    # 初始化网络
    net = get_net()

    # 处理数据
    end_points, cloud_o3d = get_and_process_data(data_dir)

    # 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # 碰撞检测
    if COLLISION_THRESH > 0:
        gg = collision_detection(gg, np.asarray(cloud_o3d.points))

    # 打印抓取位姿
    print_grasp_poses(gg)

    # 可视化
    gg.nms().sort_by_score()
    gg = gg[:50]  # 取前50个抓取可视化
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud_o3d, *grippers])


if __name__ == '__main__':
    demo(DATA_DIR)