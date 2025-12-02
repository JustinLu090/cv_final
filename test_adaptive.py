#!/usr/bin/env python3
"""
test_adaptive.py - Adaptive Memory Injection 測試腳本
=====================================================

功能:
- 測試自適應記憶注入系統
- 不需要預知遮擋時間點
- 自動檢測異常幀並注入記憶

使用方式:
    # 純測試（不加遮擋，看檢測能力）
    python test_adaptive.py --scene_dir /path/to/scene
    
    # 加入人工遮擋來測試
    python test_adaptive.py --scene_dir /path/to/scene --add_occlusion
    
    # 指定遮擋範圍
    python test_adaptive.py --add_occlusion --occlusion_start 15 --occlusion_end 25
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import json
from PIL import Image
from datetime import datetime

from utils.occlusion_tester import OcclusionTester
from utils.memory_utils import AdaptiveMemoryBuffer


class Logger:
    """同時輸出到終端和檔案的日誌器，檔案保存完整內容"""
    
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        self.log_file = None
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.log_file = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def write_full(self, message):
        """只寫入檔案，不輸出到終端（用於保存完整內容）"""
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def write_terminal_only(self, message):
        """只輸出到終端，不寫入檔案"""
        self.terminal.write(message)
    
    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()


# 全域 logger 參考，用於在函數中存取
_current_logger = None

def print_response(label, response, max_terminal_lines=3, line_width=70):
    """
    打印回應，終端截斷顯示，檔案保存完整內容
    
    Args:
        label: 標籤（如 "🎯 GT (原圖)"）
        response: 回應文字
        max_terminal_lines: 終端最多顯示幾行
        line_width: 每行最大字元數
    """
    global _current_logger
    
    if not response:
        print(f"           {label}")
        print(f"                 (無回應)")
        return
    
    # 分割成多行
    lines = [response[j:j+line_width] for j in range(0, len(response), line_width)]
    
    # 檢查是否需要截斷（終端）vs 完整顯示（檔案）
    need_truncate = len(lines) > max_terminal_lines
    
    if _current_logger and _current_logger.log_file and need_truncate:
        # 有 logger 且需要截斷：終端截斷，檔案完整
        
        # 終端只顯示截斷版本
        _current_logger.write_terminal_only(f"           {label}\n")
        for line in lines[:max_terminal_lines]:
            _current_logger.write_terminal_only(f"                 {line}\n")
        _current_logger.write_terminal_only(f"                 ... (共 {len(lines)} 行，完整內容見日誌檔案)\n")
        
        # 檔案直接顯示完整內容
        _current_logger.write_full(f"           {label}\n")
        for line in lines:
            _current_logger.write_full(f"                 {line}\n")
    else:
        # 不需要截斷，或沒有 logger：正常輸出全部
        print(f"           {label}")
        for line in lines:
            print(f"                 {line}")


def setup_logging(args):
    """設定日誌記錄"""
    global _current_logger
    
    # 建立 logs 資料夾
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日誌檔案名稱
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 從輸入路徑提取名稱
    input_name = 'unknown'
    input_path = args.video or args.scene_dir
    if input_path:
        input_name = os.path.basename(input_path.rstrip('/'))
        # 移除副檔名
        if '.' in input_name:
            input_name = os.path.splitext(input_name)[0]
    
    # 組合檔案名稱
    occ_str = f"_occ{args.occlusion_start}-{args.occlusion_end}" if args.add_occlusion else "_no_occ"
    log_filename = f"{timestamp}_{input_name}{occ_str}_{args.injection_method}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 設定 Logger
    logger = Logger(log_path)
    sys.stdout = logger
    _current_logger = logger  # 設定全域參考
    
    print(f"📝 日誌檔案: {log_path}")
    
    return logger, log_path


def save_experiment_summary(log_dir, log_path, args, results, final_status):
    """保存實驗摘要為 JSON"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'log_file': log_path,
        'config': {
            'input': args.video or args.scene_dir,
            'add_occlusion': args.add_occlusion,
            'occlusion_start': args.occlusion_start,
            'occlusion_end': args.occlusion_end,
            'occlusion_ratio': args.occlusion_ratio,
            'occlusion_type': args.occlusion_type,
            'injection_method': args.injection_method,
            'memory_size': args.memory_size,
            'max_frames': args.max_frames,
        },
        'results': {
            'total_frames': len(results),
            'anomalies_detected': len([r for r in results if r['is_anomaly']]),
            'successful_injections': len([r for r in results if r['injection'] and 'response' in r.get('injection', {})]),
            'memory_buffer_size': final_status['size'],
        },
        'frames': []
    }
    
    # 加入每幀的詳細資訊
    for r in results:
        frame_info = {
            'frame': r['frame'],
            'quality': r['quality'],
            'anomaly_score': r['anomaly_score'],
            'image_occlusion': r['image_occlusion'],
            'is_anomaly': r['is_anomaly'],
            'artificial_occlusion': r['artificial'],
        }
        if r['injection']:
            frame_info['injection'] = {
                'strength': r['injection'].get('strength'),
                'memory_frame': r['injection'].get('memory_frame'),
                'scene_match': r['injection'].get('scene_match'),
                'response': r['injection'].get('response', '')[:200],  # 截斷
                'gt_response': r['injection'].get('gt_response', '')[:200],
                'occluded_response': r['injection'].get('occluded_response', '')[:200],
            }
        summary['frames'].append(frame_info)
    
    # 計算檢測統計
    if args.add_occlusion:
        artificial = [r for r in results if r['artificial']]
        detected = [r for r in artificial if r['is_anomaly']]
        non_occluded = [r for r in results if not r['artificial']]
        false_positives = [r for r in non_occluded if r['is_anomaly']]
        
        summary['detection_stats'] = {
            'artificial_occlusion_frames': len(artificial),
            'correctly_detected': len(detected),
            'detection_rate': len(detected) / max(len(artificial), 1),
            'false_positives': len(false_positives),
            'false_positive_rate': len(false_positives) / max(len(non_occluded), 1),
        }
    
    # 保存 JSON
    json_path = log_path.replace('.log', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 實驗摘要已保存: {json_path}")
    
    return json_path


def find_model_path():
    """尋找模型路徑"""
    possible_paths = [
        'checkpoints_unified/best_unified_model.pt',
        'checkpoints/unified_model_best.pth',
        'best_unified_model.pt',
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None


def find_scene_dir():
    """尋找場景目錄"""
    possible_dirs = [
        'data/scannet/scene0011_00',
        'scannet_data/scannet_frames_test/scene0011_00',
        'scannet_data/scannet_frames_25k/scene0011_00',
    ]
    for d in possible_dirs:
        if os.path.exists(d):
            return d
    return None


def find_best_scene(data_root, split='test', min_frames=30):
    """
    尋找 frame 數最多的場景
    
    Args:
        data_root: 資料根目錄
        split: 'test' 或 'train'
        min_frames: 最少需要的 frame 數
    
    Returns:
        最佳場景的路徑，如果找不到則返回 None
    """
    from pathlib import Path
    
    data_root = Path(data_root)
    
    if split == 'test':
        scene_root = data_root / 'scannet_frames_test'
    elif split == 'train':
        scene_root = data_root / 'scannet_frames_25k'
    else:
        scene_root = data_root
    
    if not scene_root.exists():
        print(f"❌ 目錄不存在: {scene_root}")
        return None
    
    all_scene_dirs = [d for d in scene_root.iterdir() if d.is_dir()]
    
    if not all_scene_dirs:
        print(f"❌ 找不到場景: {scene_root}")
        return None
    
    # 計算每個場景的 frame 數量
    print(f"\n📊 分析 {len(all_scene_dirs)} 個場景的 frame 數量...")
    scene_frame_counts = []
    for scene_dir in all_scene_dirs:
        color_dir = scene_dir / 'color'
        if color_dir.exists():
            frame_count = len(list(color_dir.glob('*.jpg')))
        else:
            frame_count = len(list(scene_dir.glob('*.jpg'))) + len(list(scene_dir.glob('*.png')))
        
        if frame_count >= min_frames:
            scene_frame_counts.append((scene_dir, frame_count))
    
    if not scene_frame_counts:
        print(f"❌ 沒有場景有足夠的 frame (最少需要 {min_frames})")
        return None
    
    # 按 frame 數量降序排序
    scene_frame_counts.sort(key=lambda x: x[1], reverse=True)
    
    best_scene, best_count = scene_frame_counts[0]
    print(f"✅ 選擇最佳場景: {best_scene.name} ({best_count} frames)")
    
    # 顯示 top 5
    print(f"\n📊 Top 5 場景:")
    for scene_dir, frame_count in scene_frame_counts[:5]:
        marker = "→" if scene_dir == best_scene else " "
        print(f"   {marker} {scene_dir.name}: {frame_count} frames")
    
    return str(best_scene)


def load_frames(scene_dir, max_frames=40):
    """
    載入場景幀
    
    支援:
    1. 圖像目錄 (jpg, png, jpeg)
    2. 影片檔案 (mp4, avi, mov, mkv)
    """
    # 檢查是否為影片檔案
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    if os.path.isfile(scene_dir) and any(scene_dir.lower().endswith(ext) for ext in video_extensions):
        return load_frames_from_video(scene_dir, max_frames)
    
    # 圖像目錄模式
    patterns = [
        f'{scene_dir}/frame-*-color.render.jpg',
        f'{scene_dir}/frame-*.jpg',
        f'{scene_dir}/frame-*.png',
        f'{scene_dir}/*.jpg',
        f'{scene_dir}/*.jpeg',
        f'{scene_dir}/*.png',
        f'{scene_dir}/color/*.jpg',
        f'{scene_dir}/color/*.png',
        f'{scene_dir}/images/*.jpg',
        f'{scene_dir}/images/*.png',
    ]
    
    print(f"   搜尋目錄: {scene_dir}")
    print(f"   目錄是否存在: {os.path.exists(scene_dir)}")
    
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if len(files) >= 5:
            print(f"   ✅ 找到 {len(files)} 個圖像文件")
            return files[:max_frames]
    
    # 如果都找不到，列出目錄內容
    print(f"   找不到足夠的幀，目錄內容:")
    if os.path.exists(scene_dir):
        for item in os.listdir(scene_dir)[:20]:
            print(f"      {item}")
    else:
        print(f"   目錄不存在: {scene_dir}")
    
    return []


def load_frames_from_video(video_path, max_frames=40, sample_fps=2):
    """
    從影片檔案載入幀
    
    Args:
        video_path: 影片檔案路徑
        max_frames: 最大幀數
        sample_fps: 採樣頻率（每秒取幾幀）
    
    Returns:
        frames: 幀路徑列表（臨時保存的圖像）
    """
    import tempfile
    
    print(f"   📹 載入影片: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ 無法開啟影片: {video_path}")
        return []
    
    # 取得影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"   影片資訊: {total_frames} 幀, {fps:.1f} FPS, {duration:.1f} 秒")
    
    # 計算採樣間隔
    sample_interval = max(1, int(fps / sample_fps))
    
    # 建立臨時目錄儲存幀
    temp_dir = tempfile.mkdtemp(prefix='video_frames_')
    print(f"   臨時目錄: {temp_dir}")
    
    frame_paths = []
    frame_idx = 0
    saved_count = 0
    
    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 採樣
        if frame_idx % sample_interval == 0:
            frame_path = os.path.join(temp_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"   ✅ 從影片提取 {len(frame_paths)} 幀")
    
    return frame_paths


def main():
    parser = argparse.ArgumentParser(description='Test Adaptive Memory Injection')
    parser.add_argument('--model_path', type=str, default=None, help='UnifiedTempoVLM 模型路徑')
    parser.add_argument('--scene_dir', type=str, default=None, 
                       help='場景目錄或影片檔案路徑 (支援 mp4/avi/mov/mkv)')
    parser.add_argument('--video', type=str, default=None,
                       help='影片檔案路徑 (與 --scene_dir 相同功能)')
    parser.add_argument('--data_root', type=str, default=None,
                       help='ScanNet 資料根目錄（用於自動選擇最佳場景）')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'train'],
                       help='使用 test 或 train 資料集')
    parser.add_argument('--auto_best', action='store_true',
                       help='自動選擇 frame 數最多的場景')
    parser.add_argument('--max_frames', type=int, default=40, help='最大處理幀數')
    parser.add_argument('--sample_fps', type=float, default=2, help='影片採樣頻率 (每秒取幾幀)')
    parser.add_argument('--add_occlusion', action='store_true', help='人工加入遮擋')
    parser.add_argument('--occlusion_start', type=int, default=15)
    parser.add_argument('--occlusion_end', type=int, default=25)
    parser.add_argument('--occlusion_ratio', type=float, default=0.4)
    parser.add_argument('--occlusion_type', type=str, default='black',
                       choices=['black', 'white', 'noise', 'blur', 'color'],
                       help='遮擋類型: black/white/noise/blur/color')
    parser.add_argument('--injection_method', type=str, default='full',
                       choices=['raw', 'full', 'strong', 'adaptive'],
                       help='注入方法: raw(保守)/full(全圖)/strong(強力中心)/adaptive(自適應)')
    parser.add_argument('--memory_size', type=int, default=8, help='記憶庫大小')
    parser.add_argument('--anomaly_threshold', type=float, default=0.25)
    parser.add_argument('--save_results', type=str, default=None, help='儲存結果到指定目錄')
    parser.add_argument('--no_log', action='store_true', help='不記錄日誌')
    args = parser.parse_args()
    
    # 設定日誌記錄
    logger = None
    log_path = None
    if not args.no_log:
        logger, log_path = setup_logging(args)
    
    results = []
    final_status = {'size': 0}
    
    try:
        _run_experiment(args, results, final_status)
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 保存實驗摘要並關閉日誌
        if logger and log_path:
            try:
                if results:  # 只在有結果時保存
                    save_experiment_summary('logs', log_path, args, results, final_status)
            except Exception as e:
                print(f"⚠️ 保存摘要失敗: {e}")
            
            # 關閉 logger，恢復標準輸出
            sys.stdout = logger.terminal
            logger.close()
            print(f"\n✅ 實驗完成！日誌已保存到: {log_path}")


def _run_experiment(args, results, final_status_ref):
    """實際執行實驗的內部函數"""
    print("=" * 70)
    print("🧠 Adaptive Memory Injection 測試")
    print("=" * 70)
    print("\n這個模式自動：")
    print("  1. 建立高品質幀的記憶庫")
    print("  2. 檢測異常/遮擋幀")
    print("  3. 動態注入記憶特徵")
    
    # 初始化
    model_path = args.model_path or find_model_path()
    tester = OcclusionTester(unified_model_path=model_path)
    
    # 場景或影片 - 支援自動選擇最佳場景
    input_path = args.video or args.scene_dir
    
    # 如果啟用 auto_best 且有 data_root，自動選擇最佳場景
    if args.auto_best and args.data_root:
        print(f"\n🔍 自動尋找最佳場景 (frame 數最多)...")
        input_path = find_best_scene(args.data_root, split=args.split, min_frames=args.max_frames)
    
    # 如果還是沒有指定，使用預設搜尋
    if not input_path:
        input_path = find_scene_dir()
    
    if not input_path:
        print("❌ 請指定場景目錄或影片檔案")
        print("   使用方式:")
        print("   python test_adaptive.py --scene_dir /path/to/images/")
        print("   python test_adaptive.py --video /path/to/video.mp4")
        print("   python test_adaptive.py --data_root /path/to/scannet --auto_best")
        return
    
    # 載入幀
    frame_files = load_frames(input_path, max_frames=args.max_frames)
    if len(frame_files) < 5:
        print(f"❌ 幀數不足 (需要 >= 5, 找到 {len(frame_files)})")
        return
    
    print(f"\n✅ 載入 {len(frame_files)} 幀 from {input_path}")
    
    if args.add_occlusion:
        print(f"🔲 人工遮擋: Frame {args.occlusion_start}-{args.occlusion_end}")
    
    # 初始化記憶緩衝區
    memory_buffer = AdaptiveMemoryBuffer(
        max_size=args.memory_size,
        anomaly_threshold=args.anomaly_threshold
    )
    
    # 清空時序緩衝區（新場景）
    if hasattr(tester, 'clear_temporal_buffer'):
        tester.clear_temporal_buffer()
        print("🔄 時序緩衝區已清空")
    
    # 顯示是否使用 Adapter
    if tester.unified_model is not None:
        print("🧠 Adapter 時序增強: 啟用")
    else:
        print("⚠️ Adapter 時序增強: 未啟用 (使用原始 VLM 特徵)")
    
    print("\n" + "=" * 70)
    print("📊 逐幀處理")
    print("=" * 70)
    
    results = []
    
    for i, frame_file in enumerate(frame_files):
        original_img = Image.open(frame_file).convert('RGB')
        original_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        
        # 是否加入人工遮擋
        is_artificial = False
        if args.add_occlusion and args.occlusion_start <= i < args.occlusion_end:
            h, w = original_cv.shape[:2]
            cx, cy = w // 2, h // 2
            size = int(min(w, h) * args.occlusion_ratio / 2)
            
            # 根據遮擋類型應用不同的遮擋
            occ_type = args.occlusion_type
            if occ_type == 'black':
                # 黑色方塊
                cv2.rectangle(original_cv, (cx-size, cy-size), (cx+size, cy+size), (0, 0, 0), -1)
            elif occ_type == 'white':
                # 白色方塊
                cv2.rectangle(original_cv, (cx-size, cy-size), (cx+size, cy+size), (255, 255, 255), -1)
            elif occ_type == 'noise':
                # 隨機噪聲
                noise = np.random.randint(0, 255, (size*2, size*2, 3), dtype=np.uint8)
                original_cv[cy-size:cy+size, cx-size:cx+size] = noise
            elif occ_type == 'blur':
                # 高斯模糊
                roi = original_cv[cy-size:cy+size, cx-size:cx+size]
                blurred = cv2.GaussianBlur(roi, (99, 99), 0)
                original_cv[cy-size:cy+size, cx-size:cx+size] = blurred
            elif occ_type == 'color':
                # 隨機純色方塊
                random_color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(original_cv, (cx-size, cy-size), (cx+size, cy+size), random_color, -1)
            
            input_img = Image.fromarray(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
            is_artificial = True
            if i == args.occlusion_start:
                print(f"   🔲 遮擋已套用: Frame {args.occlusion_start}-{args.occlusion_end-1}, type={occ_type}, ratio={args.occlusion_ratio}")
        else:
            input_img = original_img
        
        # 提取特徵
        feat = tester.extract_features(input_img)
        adapter_meta = getattr(tester, 'last_adapter_meta', None)
        
        # 嘗試加入記憶庫
        result = memory_buffer.add_frame(feat, i, input_img, adapter_meta=adapter_meta)
        if len(result) == 5:
            added, quality, anomaly_score, is_anomaly, debug_info = result
        else:
            added, quality, anomaly_score, is_anomaly = result
            debug_info = {'image_occlusion': 0.0}
        
        img_occ = debug_info.get('image_occlusion', 0.0)
        
        # 如果異常，嘗試注入
        injection_result = None
        gt_response = None  # Ground Truth 回應
        occluded_response = None  # 遮擋圖像的直接回應（無注入）
        
        if is_anomaly and len(memory_buffer.features) > 0:
            # === 新增：提取邊緣特徵用於場景變化檢測 ===
            edge_feat = None
            try:
                edge_feat = tester.extract_edge_features(input_img)
            except Exception as e:
                pass  # 如果失敗，繼續使用 None
            
            # 使用邊緣特徵選擇最佳記憶
            best_memory, score, info = memory_buffer.get_best_memory(feat, i, edge_feat=edge_feat)
            
            if best_memory is not None:
                # 取得場景匹配度與 Adapter 可靠度
                scene_match = info.get('scene_match', 1.0)
                adapter_reliability = 1.0
                if info.get('adapter_meta'):
                    mq = info['adapter_meta'].get('memory_quality')
                    if mq is not None:
                        adapter_reliability = max(0.0, min(1.0, float(mq)))
                
                # 傳遞 scene_match 以動態調整注入強度
                base_strength = memory_buffer.compute_injection_strength(
                    anomaly_score, score, 
                    image_occlusion=img_occ,
                    scene_match=scene_match,
                    memory_reliability=adapter_reliability
                )
                
                # === 強度限制（放寬上限，中心遮罩已降低風險）===
                if args.injection_method == 'full':
                    # full 僅作用中心遮罩，允許更高上限
                    strength = min(0.35, base_strength * 0.8)
                elif args.injection_method in ('adaptive', 'strong'):
                    strength = min(0.40, base_strength * 0.9)
                else:  # raw
                    strength = min(0.45, base_strength)
                
                prompt = "Describe what you see in the center of this image."
                
                try:
                    # === 1. 注入後的回應 ===
                    response = tester.generate_with_direct_injection(
                        current_image=input_img,
                        enhanced_feat=best_memory,
                        prompt=prompt,
                        injection_method=args.injection_method,
                        injection_strength=strength
                    )
                    
                    # === 2. Ground Truth：對原始圖像（無遮擋）生成描述 ===
                    gt_response = tester.generate_description(original_img, prompt)
                    
                    # === 3. 遮擋圖像的直接回應（無注入，用於對比）===
                    occluded_response = tester.generate_description(input_img, prompt)
                    
                    injection_result = {
                        'response': response,
                        'strength': strength,
                        'memory_frame': info['timestamp'],
                        'memory_score': score,
                        'scene_match': scene_match,
                        'memory_quality': adapter_reliability,
                        'gt_response': gt_response,
                        'occluded_response': occluded_response
                    }
                except Exception as e:
                    injection_result = {'error': str(e)}
        
        results.append({
            'frame': i,
            'quality': quality,
            'anomaly_score': anomaly_score,
            'image_occlusion': img_occ,
            'is_anomaly': is_anomaly,
            'added': added,
            'artificial': is_artificial,
            'injection': injection_result
        })
        
        # 打印（包含圖像遮擋分數）
        icon = "⚠️" if is_anomaly else ("✅" if added else "➖")
        status = memory_buffer.get_status()
        occ_mark = "[OCC]" if is_artificial else "     "
        
        print(f"\n  Frame {i:2d}: {occ_mark} {icon} q={quality:.2f} a={anomaly_score:.2f} img={img_occ:.2f} mem={status['size']}", end="")
        
        if is_anomaly and injection_result:
            if 'response' in injection_result:
                resp = injection_result['response']
                gt_resp = injection_result.get('gt_response', '')
                occ_resp = injection_result.get('occluded_response', '')
                scene_match = injection_result.get('scene_match', 1.0)
                
                # 顯示場景匹配度
                scene_indicator = "🔴" if scene_match < 0.5 else ("🟡" if scene_match < 0.7 else "🟢")
                print(f" → 注入 (s={injection_result['strength']:.2f}, F{injection_result['memory_frame']}, {scene_indicator}match={scene_match:.2f})")
                
                # === 使用 print_response 顯示三種回應 ===
                # 終端截斷顯示，檔案保存完整內容
                print_response("┌─ 🎯 GT (原圖):", gt_resp, max_terminal_lines=3)
                print_response("├─ ❌ 遮擋 (無注入):", occ_resp, max_terminal_lines=3)
                print_response("└─ 💉 注入後:", resp, max_terminal_lines=3)
                    
            elif 'error' in injection_result:
                print(f" → 注入失敗: {injection_result['error'][:50]}")
            else:
                print(f" → 注入 (無回應)")
        else:
            print()
    
    # 總結
    print("\n" + "=" * 70)
    print("📊 總結")
    print("=" * 70)
    
    anomalies = [r for r in results if r['is_anomaly']]
    injected = [r for r in results if r['injection'] and 'response' in r.get('injection', {})]
    
    print(f"\n總幀數: {len(results)}")
    print(f"檢測到異常: {len(anomalies)}")
    print(f"成功注入: {len(injected)}")
    
    final_status = memory_buffer.get_status()
    print(f"記憶庫大小: {final_status['size']}")
    
    if 'zscore_stats' in final_status:
        zs = final_status['zscore_stats']
        print(f"\n📈 Z-Score 統計:")
        print(f"   歷史樣本數: {zs['history_size']}")
        print(f"   異常分數 μ: {zs['anomaly_mu']:.4f}")
        print(f"   異常分數 σ: {zs['anomaly_sigma']:.4f}")
        print(f"   動態閾值: {zs['dynamic_threshold']:.4f}")
    
    if args.add_occlusion:
        artificial = [r for r in results if r['artificial']]
        detected = [r for r in artificial if r['is_anomaly']]
        
        non_occluded = [r for r in results if not r['artificial']]
        false_positives = [r for r in non_occluded if r['is_anomaly']]
        
        print(f"\n🎯 檢測結果:")
        print(f"   人工遮擋: {len(artificial)} 幀")
        print(f"   檢測成功: {len(detected)} ({len(detected)/max(len(artificial),1)*100:.1f}%)")
        print(f"   誤報數: {len(false_positives)} ({len(false_positives)/max(len(non_occluded),1)*100:.1f}%)")
        
        # === 詳細對比分析 ===
        print(f"\n" + "=" * 70)
        print("📝 詳細對比分析")
        print("=" * 70)
        
        for r in results:
            if r['injection'] and 'response' in r['injection']:
                inj = r['injection']
                print(f"\n📍 Frame {r['frame']} (遮擋率: {r['image_occlusion']:.2f})")
                print(f"   注入強度: {inj['strength']:.2f}, 記憶來源: F{inj['memory_frame']}")
                print(f"   場景匹配: {inj.get('scene_match', 1.0):.2f}")
                
                # 使用 print_response 顯示完整內容到檔案
                gt = inj.get('gt_response', '(無)')
                occ = inj.get('occluded_response', '(無)')
                resp = inj.get('response', '(無)')
                
                print_response(" Ground Truth (原圖):", gt, max_terminal_lines=5)
                print_response(" 遮擋圖 (無注入):", occ, max_terminal_lines=5)
                print_response(" 注入後:", resp, max_terminal_lines=5)
                print("-" * 70)
    
    final_status_ref.update(final_status)


if __name__ == '__main__':
    main()
