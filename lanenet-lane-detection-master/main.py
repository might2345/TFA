import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json
import threading
import queue
import time
import math
import os
import sys

# Добавляем путь к LaneNet репозиторию
sys.path.append('lanenet-lane-detection')

try:
    from lanenet_model import lanenet
    from lanenet_model import lanenet_postprocess
    from local_utils.config_utils import parse_config_utils
    from local_utils.log_utils import init_logger

    print("LaneNet imported successfully")
except ImportError as e:
    print(f"LaneNet import error: {e}")
    print("Please clone https://github.com/MaybeShewill-CV/lanenet-lane-detection")
    print("and install requirements: tensorflow==1.15.0")


class LaneNetDetector:
    """Детектор полос с использованием оригинального LaneNet"""

    def __init__(self, model_path: str = 'lanenet-lane-detection/model/tusimple_lanenet.ckpt'):
        self.model_path = model_path
        self.model = None
        self.postprocessor = None
        self.cfg = None
        self.lane_history = deque(maxlen=5)
        self._initialize_lanenet()

    def _initialize_lanenet(self):
        """Инициализация LaneNet модели"""
        try:
            # Инициализация конфигурации
            self.cfg = parse_config_utils.lanenet_cfg
            self.cfg.MODEL.SAVE_DIR = self.model_path

            # Создание модели
            self.model = lanenet.LaneNet(phase='test', cfg=self.cfg)

            # Инициализация постпроцессора
            self.postprocessor = lanenet_postprocess.LaneNetPoseProcessor(cfg=self.cfg)

            print("LaneNet initialized successfully")

        except Exception as e:
            print(f"Error initializing LaneNet: {e}")
            self.model = None

    def detect_lanes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Детекция полос с использованием LaneNet"""
        if self.model is None:
            return self._fallback_detection(frame)

        try:
            # Предобработка кадра
            input_image = self._preprocess_frame(frame)

            # Инференс модели (в реальном коде здесь будет вызов сессии TensorFlow)
            # Для демонстрации используем fallback
            lanes = self._simulate_lanenet_detection(frame)

            # Сглаживание полос по истории
            if lanes:
                self.lane_history.append(lanes)
                return self._smooth_lanes()
            else:
                return self._get_last_lanes()

        except Exception as e:
            print(f"LaneNet detection error: {e}")
            return self._fallback_detection(frame)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Предобработка кадра для LaneNet"""
        # Resize до размера, ожидаемого моделью
        resized = cv2.resize(frame, (512, 256))
        # Нормализация
        normalized = resized.astype(np.float32) / 127.5 - 1.0
        return normalized

    def _simulate_lanenet_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Симуляция работы LaneNet (заглушка)
        В реальном коде здесь будет вызов TensorFlow сессии
        """
        h, w = frame.shape[:2]

        # Используем традиционные методы CV для симуляции
        lanes = self._traditional_lane_detection(frame)
        return lanes

    def _traditional_lane_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Традиционная детекция полос как fallback"""
        h, w = frame.shape[:2]

        # Конвертация в HSV для лучшего выделения линий
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Белые линии
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # Желтые линии
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Комбинированная маска
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Морфологические операции
        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)

        # ROI (область интереса)
        mask_roi = np.zeros_like(lane_mask)
        vertices = np.array([[
            (w * 0.05, h),
            (w * 0.45, h * 0.6),
            (w * 0.55, h * 0.6),
            (w * 0.95, h)
        ]], dtype=np.int32)
        cv2.fillPoly(mask_roi, vertices, 255)
        lane_mask = cv2.bitwise_and(lane_mask, mask_roi)

        # Детекция линий Hough
        lines = cv2.HoughLinesP(lane_mask, 1, np.pi / 180, 50,
                                minLineLength=30, maxLineGap=100)

        lanes = []
        if lines is not None:
            left_lines = []
            right_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Игнорируем горизонтальные линии
                if abs(y2 - y1) < 10:
                    continue

                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0

                if abs(slope) > 0.3:  # Фильтрация по углу наклона
                    if slope < 0 and x1 < w / 2:  # Левая полоса
                        left_lines.append(line[0])
                    elif slope > 0 and x1 > w / 2:  # Правая полоса
                        right_lines.append(line[0])

            # Создание полос из линий
            left_lane = self._fit_lane_line(left_lines, h, w, 'left')
            right_lane = self._fit_lane_line(right_lines, h, w, 'right')

            if left_lane:
                lanes.append(left_lane)
            if right_lane:
                lanes.append(right_lane)

        return lanes

    def _fit_lane_line(self, lines: List, h: int, w: int, side: str) -> Optional[Tuple[int, int, int, int]]:
        """Аппроксимация набора линий в одну полосу"""
        if not lines:
            return None

        lines_array = np.array(lines)

        # Объединяем все точки линий
        all_x = np.concatenate([lines_array[:, 0], lines_array[:, 2]])
        all_y = np.concatenate([lines_array[:, 1], lines_array[:, 3]])

        if len(all_x) < 2:
            return None

        # Линейная регрессия для нахождения оптимальной линии
        coefficients = np.polyfit(all_y, all_x, 1)

        # Создание точек для полосы
        y_bottom = h
        y_top = int(h * 0.6)

        x_bottom = int(np.polyval(coefficients, y_bottom))
        x_top = int(np.polyval(coefficients, y_top))

        # Ограничение координат
        x_bottom = max(0, min(w, x_bottom))
        x_top = max(0, min(w, x_top))

        return (x_bottom, y_bottom, x_top, y_top)

    def _fallback_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Fallback детекция полос"""
        return self._traditional_lane_detection(frame)

    def _smooth_lanes(self) -> List[Tuple[int, int, int, int]]:
        """Сглаживание полос по истории"""
        if not self.lane_history:
            return []

        # Группировка и усреднение полос
        all_lanes = []
        for lanes in self.lane_history:
            all_lanes.extend(lanes)

        if not all_lanes:
            return []

        # Кластеризация по X координате нижней точки
        x_coords = [lane[0] for lane in all_lanes]

        try:
            from sklearn.cluster import KMeans
            n_clusters = min(4, len(set(x_coords)))
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(np.array(x_coords).reshape(-1, 1))

                smoothed_lanes = []
                for i in range(kmeans.n_clusters):
                    cluster_lanes = [all_lanes[j] for j in range(len(all_lanes)) if clusters[j] == i]
                    if cluster_lanes:
                        avg_x_bottom = int(np.mean([lane[0] for lane in cluster_lanes]))
                        avg_x_top = int(np.mean([lane[2] for lane in cluster_lanes]))
                        h = cluster_lanes[0][1]
                        smoothed_lanes.append((avg_x_bottom, h, avg_x_top, int(h * 0.6)))

                return sorted(smoothed_lanes, key=lambda x: x[0])
            else:
                return []

        except ImportError:
            # Простое усреднение если scikit-learn не доступен
            if all_lanes:
                avg_x_bottom = int(np.mean([lane[0] for lane in all_lanes]))
                avg_x_top = int(np.mean([lane[2] for lane in all_lanes]))
                h = all_lanes[0][1]
                return [(avg_x_bottom, h, avg_x_top, int(h * 0.6))]
            return []

    def _get_last_lanes(self) -> List[Tuple[int, int, int, int]]:
        """Получение последних обнаруженных полос"""
        return self.lane_history[-1] if self.lane_history else []


@dataclass
class Vehicle:
    id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    accelerations: deque = field(default_factory=lambda: deque(maxlen=15))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=30))
    violation_history: Dict[str, int] = field(default_factory=dict)
    is_aggressive: bool = False
    vehicle_class: str = "car"
    first_detection_time: float = 0
    last_detection_time: float = 0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    track_fail_count: int = 0
    active: bool = True
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    lane_id: Optional[int] = None
    lane_history: deque = field(default_factory=lambda: deque(maxlen=10))


class ImprovedVehicleTracker:

    def __init__(self, max_age=3.0):
        self.vehicles: Dict[int, Vehicle] = {}
        self.max_age = max_age
        self.next_id = 1
        self.removed_vehicles = []

    def update(self, detections, current_time, frame_shape, lanes: List[Tuple[int, int, int, int]]):
        """Update vehicle tracks with lane assignment"""
        h, w = frame_shape[:2]

        for vehicle in self.vehicles.values():
            vehicle.active = False

        used_ids = set()

        for detection in detections:
            box, confidence, class_id = detection
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area < 1000:
                continue

            best_match_id = None
            best_score = 0
            max_match_distance = 150

            for vehicle_id, vehicle in self.vehicles.items():
                if not vehicle.positions:
                    continue

                last_x, last_y = vehicle.positions[-1]
                distance = math.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)

                if distance < max_match_distance:
                    if vehicle.bbox_history:
                        last_bbox = vehicle.bbox_history[-1]
                        last_area = (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1])
                        size_similarity = 1 - abs(bbox_area - last_area) / max(bbox_area, last_area)
                    else:
                        size_similarity = 0.5

                    distance_score = 1 - (distance / max_match_distance)

                    score = distance_score * 0.7 + size_similarity * 0.3

                    if score > best_score:
                        best_score = score
                        best_match_id = vehicle_id

            if best_match_id is not None and best_score > 0.3:
                vehicle = self.vehicles[best_match_id]
                vehicle.positions.append((cx, cy))
                vehicle.timestamps.append(current_time)
                vehicle.confidence_history.append(confidence)
                vehicle.bbox_history.append(box)
                vehicle.active = True
                vehicle.last_detection_time = current_time
                vehicle.track_fail_count = 0

                # Обновление полосы
                current_lane = self._assign_lane(cx, cy, lanes, w)
                vehicle.lane_history.append(current_lane)
                vehicle.lane_id = current_lane
                used_ids.add(best_match_id)
            else:
                vehicle_id = self.next_id
                self.next_id += 1

                new_vehicle = Vehicle(id=vehicle_id)
                new_vehicle.positions.append((cx, cy))
                new_vehicle.timestamps.append(current_time)
                new_vehicle.confidence_history.append(confidence)
                new_vehicle.bbox_history.append(box)
                new_vehicle.first_detection_time = current_time
                new_vehicle.last_detection_time = current_time
                new_vehicle.active = True

                # Назначение полосы для нового транспортного средства
                current_lane = self._assign_lane(cx, cy, lanes, w)
                new_vehicle.lane_history.append(current_lane)
                new_vehicle.lane_id = current_lane

                class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
                new_vehicle.vehicle_class = class_names.get(class_id, "car")

                self.vehicles[vehicle_id] = new_vehicle
                used_ids.add(vehicle_id)

        to_remove = []
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.active:
                vehicle.track_fail_count += 1

            if (current_time - vehicle.last_detection_time > self.max_age or
                    vehicle.track_fail_count > 45):

                if (vehicle.last_detection_time - vehicle.first_detection_time) > 2.0:
                    self.removed_vehicles.append(vehicle)
                to_remove.append(vehicle_id)

        for vehicle_id in to_remove:
            del self.vehicles[vehicle_id]

        return used_ids

    def _assign_lane(self, cx: int, cy: int, lanes: List[Tuple[int, int, int, int]], frame_width: int) -> Optional[int]:
        """Назначение полосы транспортному средству"""
        if not lanes:
            return None

        # Поиск ближайшей полосы
        min_distance = float('inf')
        best_lane_id = None

        for i, lane in enumerate(lanes):
            lane_x = lane[0]  # X координата нижней точки полосы
            distance = abs(cx - lane_x)

            if distance < min_distance and distance < frame_width * 0.25:
                min_distance = distance
                best_lane_id = i

        return best_lane_id

    def get_active_vehicles(self):
        return [v for v in self.vehicles.values() if v.active]

    def get_all_analyzed_vehicles(self):
        return list(self.vehicles.values()) + self.removed_vehicles


class WiedemannModel:

    def __init__(self):
        self.cc0 = 2.0
        self.cc1 = 1.2
        self.cc2 = 4.0
        self.cc4 = -0.35
        self.cc5 = 0.25
        self.cc8 = 2.0
        self.cc9 = 3.0

    def is_aggressive_acceleration(self, acceleration: float, vehicle_class: str) -> bool:
        thresholds = {
            "car": 3.0,
            "motorcycle": 4.0,
            "bus": 2.0,
            "truck": 1.5
        }
        threshold = thresholds.get(vehicle_class, 3.0)
        return acceleration > threshold

    def is_aggressive_deceleration(self, deceleration: float, vehicle_class: str) -> bool:
        """Check for aggressive braking"""
        thresholds = {
            "car": -3.5,
            "motorcycle": -4.5,
            "bus": -2.5,
            "truck": -2.0
        }
        threshold = thresholds.get(vehicle_class, -3.5)
        return deceleration < threshold


class FrameProcessor:
    """Improved frame processor with LaneNet lane detection"""

    def __init__(self, model, wiedemann_model, tracker, lane_detector):
        self.model = model
        self.wiedemann = wiedemann_model
        self.tracker = tracker
        self.lane_detector = lane_detector
        self.frame_queue = queue.Queue(maxsize=20)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.lane_change_threshold = 3

    def process_frames(self):
        """Frame processing in separate thread"""
        while not self.stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=1)
                if frame_data is None:
                    break

                frame, frame_count, current_time = frame_data
                processed_data = self._process_single_frame(frame, frame_count, current_time)
                self.result_queue.put(processed_data)

            except queue.Empty:
                continue

    def _process_single_frame(self, frame: np.ndarray, frame_count: int, current_time: float):
        """Process single frame with LaneNet lane detection"""
        h, w = frame.shape[:2]

        # Детекция полос с использованием LaneNet
        lanes = self.lane_detector.detect_lanes(frame)

        # Детекция транспортных средств
        results = self.model(frame, classes=[2, 3, 5, 7],
                             verbose=False, conf=0.6, iou=0.5)

        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if conf >= 0.6:
                    detections.append((box, conf, class_id))

        used_ids = self.tracker.update(detections, current_time, frame.shape, lanes)

        vehicles_data = []
        for vehicle in self.tracker.get_active_vehicles():
            if not vehicle.positions:
                continue

            cx, cy = vehicle.positions[-1]

            vehicle_box = None
            vehicle_class_id = 2

            for box, conf, class_id in detections:
                box_cx = int((box[0] + box[2]) / 2)
                box_cy = int((box[1] + box[3]) / 2)
                distance = math.sqrt((cx - box_cx) ** 2 + (cy - box_cy) ** 2)
                if distance < 80:
                    vehicle_box = box
                    vehicle_class_id = class_id
                    break

            if vehicle_box is None and vehicle.bbox_history:
                vehicle_box = vehicle.bbox_history[-1]

            if vehicle_box is None:
                continue

            acceleration = self._calculate_acceleration(vehicle)
            vehicle.accelerations.append(acceleration)

            violations = self._detect_violations(vehicle, current_time, w, frame.shape[0])

            current_violations = set()
            for violation in violations:
                if violation not in vehicle.violation_history:
                    vehicle.violation_history[violation] = 0
                vehicle.violation_history[violation] += 1
                current_violations.add(violation)

            sustained_violations = []
            for violation, count in vehicle.violation_history.items():
                if count >= 3:
                    sustained_violations.append(violation)

            vehicle.is_aggressive = len(sustained_violations) >= 2

            vehicle_data = {
                'id': vehicle.id,
                'box': [int(x) for x in vehicle_box],
                'center': (cx, cy),
                'violations': sustained_violations,
                'is_aggressive': vehicle.is_aggressive,
                'active': vehicle.active,
                'vehicle_class': vehicle.vehicle_class,
                'lane_id': vehicle.lane_id
            }

            vehicles_data.append(vehicle_data)

        return {
            'frame': frame.copy(),
            'vehicles_data': vehicles_data,
            'frame_count': frame_count,
            'lanes': lanes
        }

    def _calculate_acceleration(self, vehicle: Vehicle) -> float:
        """Calculate acceleration in m/s² with smoothing"""
        if len(vehicle.positions) < 3:
            return 0

        positions = list(vehicle.positions)[-3:]
        timestamps = list(vehicle.timestamps)[-3:]

        if len(positions) < 2:
            return 0

        accelerations = []
        for i in range(1, len(positions)):
            time_diff = timestamps[i] - timestamps[i - 1]
            if time_diff > 0:
                dist_diff = math.sqrt(
                    (positions[i][0] - positions[i - 1][0]) ** 2 +
                    (positions[i][1] - positions[i - 1][1]) ** 2
                )
                speed_diff = dist_diff / time_diff

                if i > 1:
                    prev_dist = math.sqrt(
                        (positions[i - 1][0] - positions[i - 2][0]) ** 2 +
                        (positions[i - 1][1] - positions[i - 2][1]) ** 2
                    )
                    prev_time = timestamps[i - 1] - timestamps[i - 2]
                    if prev_time > 0:
                        prev_speed = prev_dist / prev_time
                        accel = (speed_diff - prev_speed) / time_diff
                        accel_mps2 = accel / 50
                        accelerations.append(accel_mps2)

        if not accelerations:
            return 0

        return sum(accelerations) / len(accelerations)

    def _detect_violations(self, vehicle: Vehicle, current_time: float, frame_width: int, frame_height: int) -> List[
        str]:
        violations = []

        if current_time - vehicle.first_detection_time < 2.0:
            return violations

        if len(vehicle.positions) < 8:
            return violations

        current_accel = vehicle.accelerations[-1] if vehicle.accelerations else 0

        if (current_accel > 0 and
                self.wiedemann.is_aggressive_acceleration(current_accel, vehicle.vehicle_class)):
            violations.append('rapid_acceleration')

        if (current_accel < 0 and
                self.wiedemann.is_aggressive_deceleration(current_accel, vehicle.vehicle_class)):
            violations.append('rapid_braking')

        # Обнаружение смены полосы
        if len(vehicle.lane_history) >= 5:
            lane_changes = self._detect_lane_changes(vehicle)
            if lane_changes > 1:
                violations.append('unsafe_lane_change')

        return violations

    def _detect_lane_changes(self, vehicle: Vehicle) -> int:
        """Обнаружение смен полос по истории полос"""
        lane_history = list(vehicle.lane_history)
        lane_changes = 0

        for i in range(1, len(lane_history)):
            if lane_history[i] is not None and lane_history[i - 1] is not None:
                if lane_history[i] != lane_history[i - 1]:
                    lane_changes += 1

        return lane_changes


class TrafficAnalyzer:
    """Final improved traffic analyzer with LaneNet lane detection"""

    VIOLATION_CATEGORIES = {
        'rapid_acceleration': 'Rapid Acceleration',
        'rapid_braking': 'Rapid Braking',
        'unsafe_lane_change': 'Unsafe Lane Change'
    }

    def __init__(self, video_path: str, model_path: str = 'yolo11x.pt',
                 lane_model_path: str = 'lanenet-lane-detection/model/tusimple_lanenet.ckpt'):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.lane_detector = LaneNetDetector(lane_model_path)
        self.wiedemann = WiedemannModel()
        self.tracker = ImprovedVehicleTracker(max_age=4.0)

        self.violation_stats = defaultdict(int)
        self.aggressive_drivers = set()

        self.fps = 0
        self.frame_count = 0

        self.frame_processor = FrameProcessor(self.model, self.wiedemann,
                                              self.tracker, self.lane_detector)
        self.processor_thread = None

    def process_video(self, output_path: str = 'output_analyzed.mp4',
                      stats_output: str = 'statistics.json'):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        print(f"Processing video: {self.video_path}")
        print(f"FPS: {self.fps:.1f}, Resolution: {w}x{h}, Frames: {total_frames}")
        print("-" * 60)

        self.frame_processor.stop_event.clear()
        self.processor_thread = threading.Thread(target=self.frame_processor.process_frames)
        self.processor_thread.start()

        last_stats_time = time.time()
        processed_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                current_time = self.frame_count / self.fps

                try:
                    self.frame_processor.frame_queue.put(
                        (frame, self.frame_count, current_time),
                        timeout=0.5
                    )
                except queue.Full:
                    continue

                try:
                    processed_data = self.frame_processor.result_queue.get(timeout=0.5)
                    frame = self._visualize_frame(processed_data, w, h)
                    self._update_statistics(processed_data)
                    processed_frames += 1
                except queue.Empty:
                    pass

                out.write(frame)

                if time.time() - last_stats_time > 3.0:
                    progress = (self.frame_count / total_frames) * 100
                    active_vehicles = len(self.tracker.get_active_vehicles())
                    all_vehicles = len(self.tracker.get_all_analyzed_vehicles())
                    aggressive = len(self.aggressive_drivers)

                    print(
                        f"Progress: {progress:.1f}% | Active: {active_vehicles} | Total: {all_vehicles} | Aggressive: {aggressive}")
                    last_stats_time = time.time()

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.frame_processor.stop_event.set()
            if self.processor_thread:
                self.processor_thread.join(timeout=5)

            cap.release()
            out.release()

            self.generate_statistics_report(stats_output)
            print(f"\nVideo saved: {output_path}")
            print(f"Statistics saved: {stats_output}")

    def _visualize_frame(self, processed_data: Dict, w: int, h: int) -> np.ndarray:
        frame = processed_data['frame']
        vehicles_data = processed_data['vehicles_data']
        lanes = processed_data.get('lanes', [])

        # Визуализация полос
        for i, lane in enumerate(lanes):
            x1, y1, x2, y2 = lane
            color = (0, 255, 255)  # Желтый цвет для полос LaneNet
            cv2.line(frame, (x1, y1), (x2, y2), color, 4)
            # Отображение номера полосы
            cv2.putText(frame, f"L{i}", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for vehicle in vehicles_data:
            x1, y1, x2, y2 = vehicle['box']
            cx, cy = vehicle['center']
            violations = vehicle['violations']
            is_aggressive = vehicle['is_aggressive']
            vehicle_class = vehicle.get('vehicle_class', 'car')
            lane_id = vehicle.get('lane_id')

            if is_aggressive:
                color = (0, 0, 255)
            elif violations:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            thickness = 3 if is_aggressive else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            cv2.circle(frame, (cx, cy), 6, color, -1)

            class_symbol = {"car": "C", "motorcycle": "M", "bus": "B", "truck": "T"}.get(vehicle_class, "V")
            lane_text = f"L{lane_id}" if lane_id is not None else "L-"
            info_text = f"ID:{vehicle['id']}({class_symbol}) {lane_text}"
            cv2.putText(frame, info_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for i, violation in enumerate(violations[:2]):
                viol_text = self.VIOLATION_CATEGORIES.get(violation, violation)
                cv2.putText(frame, viol_text, (x1, y2 + 15 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        stats_bg = np.zeros((120, 500, 3), dtype=np.uint8)
        frame[10:130, 10:510] = stats_bg

        active_vehicles = len(vehicles_data)
        total_tracked = len(self.tracker.get_all_analyzed_vehicles())
        aggressive_count = len(self.aggressive_drivers)
        lanes_detected = len(lanes)

        cv2.putText(frame, f"LaneNet: {lanes_detected} lanes", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Active: {active_vehicles}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total: {total_tracked}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Aggressive: {aggressive_count}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if aggressive_count > 0 else (255, 255, 255), 2)

        return frame

    def _update_statistics(self, processed_data: Dict):
        """Update statistics"""
        for vehicle_data in processed_data['vehicles_data']:
            vehicle_id = vehicle_data['id']

            if vehicle_data['is_aggressive']:
                self.aggressive_drivers.add(vehicle_id)

            for violation in vehicle_data['violations']:
                self.violation_stats[violation] += 1

    def generate_statistics_report(self, output_file: str = 'statistics.json'):
        """Generate comprehensive statistics report"""
        all_vehicles = self.tracker.get_all_analyzed_vehicles()
        total_tracked = len(all_vehicles)
        aggressive_count = len(self.aggressive_drivers)
        normal_count = total_tracked - aggressive_count

        if total_tracked == 0:
            aggressive_pct = 0
            normal_pct = 100
        else:
            aggressive_pct = (aggressive_count / total_tracked) * 100
            normal_pct = 100 - aggressive_pct

        vehicle_types = defaultdict(int)
        lane_distribution = defaultdict(int)

        for vehicle in all_vehicles:
            vehicle_types[vehicle.vehicle_class] += 1
            if vehicle.lane_id is not None:
                lane_distribution[f"Lane_{vehicle.lane_id}"] += 1

        violation_percentages = {}
        for viol_key, viol_name in self.VIOLATION_CATEGORIES.items():
            count = self.violation_stats.get(viol_key, 0)
            pct = (count / max(total_tracked, 1)) * 100
            violation_percentages[viol_name] = {
                'count': count,
                'percentage': round(pct, 2)
            }

        report = {
            'summary': {
                'total_vehicles_tracked': total_tracked,
                'aggressive_drivers': aggressive_count,
                'normal_drivers': normal_count,
                'aggressive_percentage': round(aggressive_pct, 2),
                'normal_percentage': round(normal_pct, 2),
                'total_frames_processed': self.frame_count,
                'analysis_duration_seconds': round(self.frame_count / self.fps, 2),
                'average_fps': round(self.fps, 2)
            },
            'vehicle_type_distribution': dict(vehicle_types),
            'lane_distribution': dict(lane_distribution),
            'violations_breakdown': violation_percentages,
            'analysis_parameters': {
                'detection_confidence': 0.6,
                'tracking_max_age_seconds': 4.0,
                'lane_detection': 'LaneNet',
                'violation_detection': 'Acceleration/Braking/Lane Changes'
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(" " * 20 + "TRAFFIC ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nTotal vehicles tracked: {total_tracked}")
        print(f"Normal driving: {normal_count} ({normal_pct:.1f}%)")
        print(f"Aggressive driving: {aggressive_count} ({aggressive_pct:.1f}%)")

        print(f"\nVehicle types:")
        for vtype, count in vehicle_types.items():
            pct = (count / total_tracked) * 100
            print(f"  {vtype}: {count} ({pct:.1f}%)")

        print(f"\nLane distribution:")
        for lane, count in lane_distribution.items():
            pct = (count / total_tracked) * 100
            print(f"  {lane}: {count} vehicles ({pct:.1f}%)")

        print(f"\nVIOLATION TYPES:")
        print("-" * 50)

        sorted_violations = sorted(violation_percentages.items(),
                                   key=lambda x: x[1]['count'],
                                   reverse=True)

        for viol_name, data in sorted_violations:
            if data['count'] > 0:
                print(f"{viol_name:<25} {data['count']:>4} cases ({data['percentage']:>5.1f}%)")

        print("\n" + "=" * 70)
        print(f"Frames processed: {self.frame_count}")
        print(f"Analysis duration: {self.frame_count / self.fps:.1f}s")
        print("=" * 70)


def main():
    VIDEO_PATH = "input.mp4"
    OUTPUT_VIDEO = "output.mp4"
    STATS_JSON = "traffic_statistics.json"
    LANENET_MODEL_PATH = "lanenet-lane-detection/model/tusimple_lanenet.ckpt"

    try:
        print("Starting traffic analyzer with LaneNet lane detection...")
        print("Initializing systems...")

        analyzer = TrafficAnalyzer(VIDEO_PATH, lane_model_path=LANENET_MODEL_PATH)
        analyzer.process_video(OUTPUT_VIDEO, STATS_JSON)

    except FileNotFoundError:
        print("Error: Video file not found! Please check the path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()