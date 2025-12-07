#!/usr/bin/env python3
"""
yolo_occlusion.py - 基於 YOLO 的物件偵測遮擋工具
====================================================

使用 YOLOv8 偵測場景中的物件（椅子、桌子、沙發等），
並在指定幀中遮擋這些物件，用於測試 GRU 記憶恢復能力。

使用方式:
    from utils.yolo_occlusion import YOLOOccluder
    
    occluder = YOLOOccluder()
    occluded_img, objects = occluder.occlude_objects(
        image, 
        target_classes=['chair', 'couch', 'dining table']
    )
"""

import cv2
import numpy as np
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics 未安裝，請執行: pip install ultralytics")


class YOLOOccluder:
    """
    YOLO 物件遮擋器
    
    使用 YOLOv8 偵測物件並遮擋
    """
    
    # COCO 資料集類別名稱 (YOLOv8 預設使用)
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }
    
    # 室內常見物件（ScanNet 場景）
    INDOOR_OBJECTS = ['chair', 'couch', 'bed', 'dining table', 'potted plant', 
                      'tv', 'laptop', 'book', 'vase', 'bottle']
    
    def __init__(self, model_size='m', confidence_threshold=0.25, device='cuda'):
        """
        初始化 YOLO 遮擋器
        
        Args:
            model_size: YOLO 模型大小 ('n', 's', 'm', 'l', 'x')
                       n = nano (最快，準確度較低)
                       s = small
                       m = medium (推薦)
                       l = large
                       x = xlarge (最慢，準確度最高)
            confidence_threshold: 信心度閾值 (降低到 0.25 以偵測更多物件)
            device: 運行設備
        """
        if not YOLO_AVAILABLE:
            raise ImportError("請安裝 ultralytics: pip install ultralytics")
        
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # 載入 YOLO 模型
        model_name = f'yolov8{model_size}.pt'
        print(f"📦 載入 YOLO 模型: {model_name}")
        self.model = YOLO(model_name)
        
        # 如果有 GPU 則使用
        if device == 'cuda' and torch.cuda.is_available():
            self.model.to('cuda')
        
        print(f"✅ YOLO 模型已載入 (device: {device})")
    
    def detect_objects(self, image, target_classes=None):
        """
        偵測圖像中的物件
        
        Args:
            image: PIL Image 或 numpy array (RGB)
            target_classes: 目標類別列表，None = 偵測所有類別
        
        Returns:
            detections: list of dict, 每個包含:
                - class_name: 類別名稱
                - confidence: 信心度
                - bbox: [x1, y1, x2, y2]
                - area: 面積
        """
        # 轉換為 numpy array
        if hasattr(image, 'convert'):  # PIL Image
            img_array = np.array(image)
        else:
            img_array = image
        
        # YOLO 推論
        results = self.model(img_array, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.COCO_CLASSES.get(cls_id, 'unknown')
                
                # 過濾目標類別
                if target_classes and class_name not in target_classes:
                    continue
                
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                
                detections.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'area': int(area)
                })
        
        return detections
    
    def select_medium_object(self, detections, min_area_ratio=0.02, max_area_ratio=0.15):
        """
        從偵測結果中選擇一個中等大小的物件
        
        Args:
            detections: YOLO 偵測結果列表
            min_area_ratio: 最小面積比例（相對於圖片總面積）
            max_area_ratio: 最大面積比例
        
        Returns:
            selected_detection: 選中的物件，None 如果沒有符合條件的
        """
        if not detections:
            return None
        
        # 假設圖片大小（可以從第一個 bbox 推算）
        if detections:
            # 從 bbox 推算圖片大致尺寸
            max_x = max(d['bbox'][2] for d in detections)
            max_y = max(d['bbox'][3] for d in detections)
            image_area = max_x * max_y
        else:
            return None
        
        # 過濾中等大小的物件
        medium_objects = []
        for det in detections:
            area_ratio = det['area'] / image_area
            if min_area_ratio <= area_ratio <= max_area_ratio:
                medium_objects.append(det)
        
        if not medium_objects:
            # 如果沒有中等大小的，選擇最接近中間大小的
            sorted_dets = sorted(detections, key=lambda x: x['area'])
            if len(sorted_dets) > 0:
                mid_idx = len(sorted_dets) // 2
                return sorted_dets[mid_idx]
            return None
        
        # 隨機選擇一個中等大小的物件
        import random
        return random.choice(medium_objects)
    
    def occlude_single_object(self, image, target_classes=None, occlusion_color=(0, 0, 0),
                              min_area_ratio=0.02, max_area_ratio=0.15, random_selection=True):
        """
        偵測並遮擋單一物件（隨機選擇中等大小）
        
        Args:
            image: PIL Image 或 numpy array (RGB)
            target_classes: 目標類別列表，None = 使用預設室內物件
            occlusion_color: 遮擋顏色 (R, G, B)
            min_area_ratio: 最小面積比例
            max_area_ratio: 最大面積比例
            random_selection: 是否隨機選擇（True）或選擇最接近中間大小的（False）
        
        Returns:
            occluded_image: 遮擋後的圖像 (numpy array, RGB)
            selected_detection: 被遮擋的物件信息
            all_detections: 所有偵測到的物件
        """
        # 轉換為 numpy array
        if hasattr(image, 'convert'):  # PIL Image
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        h, w = img_array.shape[:2]
        image_area = h * w
        
        # 使用預設室內物件類別
        if target_classes is None:
            target_classes = self.INDOOR_OBJECTS
        
        # 偵測物件
        all_detections = self.detect_objects(img_array, target_classes)
        
        if not all_detections:
            return img_array, None, []
        
        # 選擇中等大小的物件
        medium_objects = []
        for det in all_detections:
            area_ratio = det['area'] / image_area
            if min_area_ratio <= area_ratio <= max_area_ratio:
                medium_objects.append(det)
        
        # 選擇要遮擋的物件
        if medium_objects:
            if random_selection:
                import random
                selected = random.choice(medium_objects)
            else:
                # 選擇最接近中間大小的
                selected = sorted(medium_objects, key=lambda x: x['area'])[len(medium_objects) // 2]
        else:
            # 如果沒有中等大小，選擇最接近中等大小的
            if not all_detections:
                return img_array, None, []
            sorted_dets = sorted(all_detections, key=lambda x: x['area'])
            selected = sorted_dets[len(sorted_dets) // 2]
        
        # 遮擋選中的物件
        x1, y1, x2, y2 = selected['bbox']
        cv2.rectangle(img_array, (x1, y1), (x2, y2), occlusion_color, -1)
        
        return img_array, selected, all_detections
    
    def occlude_multiple_objects(self, image, target_classes=None, occlusion_color=(0, 0, 0),
                                 min_area=2000, max_objects=3, size_preference='medium'):
        """
        偵測並遮擋多個物件（最多 max_objects 個）
        
        Args:
            image: PIL Image 或 numpy array (RGB)
            target_classes: 目標類別列表，None = 偵測所有物件
            occlusion_color: 遮擋顏色 (R, G, B)
            min_area: 最小物件面積（避免遮擋太小的物件）
            max_objects: 最多遮擋幾個物件
            size_preference: 物件大小偏好
                - 'medium': 選擇中等大小的物件
                - 'large': 選擇大物件
                - 'small': 選擇小物件
                - 'random': 隨機選擇
        
        Returns:
            occluded_image: 遮擋後的圖像 (numpy array, RGB)
            occluded_objects: 被遮擋的物件列表
            all_detections: 所有偵測到的物件
        """
        # 轉換為 numpy array
        if hasattr(image, 'convert'):  # PIL Image
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # 偵測物件（target_classes=None 會偵測所有物件）
        all_detections = self.detect_objects(img_array, target_classes=target_classes)
        
        if not all_detections:
            return img_array, [], []
        
        # 過濾太小的物件
        valid_detections = [d for d in all_detections if d['area'] >= min_area]
        
        if not valid_detections:
            return img_array, [], all_detections
        
        # 按面積排序
        sorted_dets = sorted(valid_detections, key=lambda x: x['area'])
        
        # 根據 size_preference 選擇物件
        # 🆕 優先選中等物件，如果沒有就往更大的找
        if size_preference == 'medium':
            # 策略：中等 (20%-80%) → 大 (80%-100%) → 全部
            n = len(sorted_dets)
            start_medium = int(n * 0.2)
            end_medium = int(n * 0.8)
            
            # 1. 先嘗試中等物件
            if end_medium > start_medium:
                candidate_objects = sorted_dets[start_medium:end_medium]
            else:
                candidate_objects = []
            
            # 2. 如果中等物件不夠，加入大物件 (80%-100%)
            if len(candidate_objects) < max_objects and end_medium < n:
                large_objects = sorted_dets[end_medium:]
                candidate_objects.extend(large_objects)
                print(f"  💡 中等物件不足，加入 {len(large_objects)} 個大物件")
            
            # 3. 如果還是不夠，加入所有物件
            if len(candidate_objects) < max_objects:
                candidate_objects = sorted_dets
                print(f"  💡 物件不足，使用所有 {len(candidate_objects)} 個物件")
                
        elif size_preference == 'large':
            # 選擇大物件（前 50%）
            n = len(sorted_dets)
            candidate_objects = sorted_dets[n//2:]
        elif size_preference == 'small':
            # 選擇小物件（後 50%）
            n = len(sorted_dets)
            candidate_objects = sorted_dets[:n//2]
        else:  # random
            candidate_objects = valid_detections
        
        # 隨機選擇最多 max_objects 個物件
        import random
        num_to_occlude = min(len(candidate_objects), max_objects)
        selected_objects = random.sample(candidate_objects, num_to_occlude)
        
        # 遮擋選中的物件
        for obj in selected_objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(img_array, (x1, y1), (x2, y2), occlusion_color, -1)
        
        return img_array, selected_objects, all_detections
    
    def occlude_objects(self, image, target_classes=None, occlusion_color=(0, 0, 0),
                       min_area=1000, max_objects=5, occlusion_type='solid'):
        """
        偵測並遮擋指定物件
        
        Args:
            image: PIL Image 或 numpy array (RGB)
            target_classes: 目標類別列表，None = 使用預設室內物件
            occlusion_color: 遮擋顏色 (R, G, B)
            min_area: 最小物件面積（避免遮擋太小的物件）
            max_objects: 最多遮擋幾個物件
            occlusion_type: 遮擋類型
                - 'solid': 純色遮擋
                - 'noise': 噪聲遮擋
                - 'blur': 模糊遮擋
        
        Returns:
            occluded_image: 遮擋後的圖像 (numpy array, RGB)
            detections: 偵測到的物件列表
        """
        # 轉換為 numpy array
        if hasattr(image, 'convert'):  # PIL Image
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # 使用預設室內物件類別
        if target_classes is None:
            target_classes = self.INDOOR_OBJECTS
        
        # 偵測物件
        detections = self.detect_objects(img_array, target_classes)
        
        # 過濾太小的物件
        detections = [d for d in detections if d['area'] >= min_area]
        
        # 按面積排序（優先遮擋大物件）
        detections.sort(key=lambda x: x['area'], reverse=True)
        
        # 限制遮擋數量
        detections = detections[:max_objects]
        
        # 遮擋物件
        occluded_count = 0
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            if occlusion_type == 'solid':
                # 純色遮擋
                cv2.rectangle(img_array, (x1, y1), (x2, y2), occlusion_color, -1)
            
            elif occlusion_type == 'noise':
                # 噪聲遮擋
                noise = np.random.randint(0, 255, (y2-y1, x2-x1, 3), dtype=np.uint8)
                img_array[y1:y2, x1:x2] = noise
            
            elif occlusion_type == 'blur':
                # 模糊遮擋
                roi = img_array[y1:y2, x1:x2]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    kernel_size = max(51, min(roi.shape[0], roi.shape[1]) // 3)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                    img_array[y1:y2, x1:x2] = blurred
            
            occluded_count += 1
        
        return img_array, detections
    
    def visualize_detections(self, image, detections, show_labels=True):
        """
        視覺化偵測結果（繪製邊界框）
        
        Args:
            image: numpy array (RGB)
            detections: 偵測結果列表
            show_labels: 是否顯示標籤
        
        Returns:
            vis_image: 視覺化圖像
        """
        img_vis = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # 繪製邊界框
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 繪製標籤
            if show_labels:
                label = f"{class_name} {confidence:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_vis, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(img_vis, label, (x1, y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_vis


def test_yolo_occlusion():
    """測試 YOLO 遮擋功能"""
    from PIL import Image
    
    print("=" * 70)
    print("測試 YOLO 物件遮擋")
    print("=" * 70)
    
    # 載入測試圖像
    test_image_path = Path("scannet_data/scannet_frames_test/scene0757_00/color/0.jpg")
    
    if not test_image_path.exists():
        print(f"❌ 測試圖像不存在: {test_image_path}")
        return
    
    image = Image.open(test_image_path).convert('RGB')
    print(f"✅ 載入圖像: {test_image_path}")
    
    # 初始化遮擋器
    occluder = YOLOOccluder(model_size='n', confidence_threshold=0.3)
    
    # 偵測物件
    print("\n📊 偵測物件...")
    detections = occluder.detect_objects(image, target_classes=YOLOOccluder.INDOOR_OBJECTS)
    
    print(f"\n✅ 偵測到 {len(detections)} 個室內物件:")
    for det in detections:
        print(f"   - {det['class_name']}: {det['confidence']:.2f} (area: {det['area']})")
    
    # 遮擋物件
    print("\n🎭 遮擋物件...")
    occluded_img, occluded_dets = occluder.occlude_objects(
        image, 
        target_classes=['chair', 'couch', 'dining table'],
        occlusion_type='solid',
        max_objects=3
    )
    
    print(f"\n✅ 已遮擋 {len(occluded_dets)} 個物件")
    
    # 保存結果
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 原圖 + 偵測框
    vis_img = occluder.visualize_detections(np.array(image), detections)
    cv2.imwrite(str(output_dir / "yolo_detections.jpg"), 
                cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    # 遮擋後
    cv2.imwrite(str(output_dir / "yolo_occluded.jpg"),
                cv2.cvtColor(occluded_img, cv2.COLOR_RGB2BGR))
    
    print(f"\n✅ 結果已保存:")
    print(f"   - {output_dir / 'yolo_detections.jpg'}")
    print(f"   - {output_dir / 'yolo_occluded.jpg'}")


if __name__ == "__main__":
    test_yolo_occlusion()
