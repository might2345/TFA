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


@dataclass
class Vehicle:
    id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    speeds: deque = field(default_factory=lambda: deque(maxlen=15))
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
    distance_traveled: float = 0
    last_speed_calc_time: float = 0
    calibrated: bool = False
    calibration_frames: int = 0


class ImprovedVehicleTracker:

    def __init__(self, max_age=3.0):
        self.vehicles: Dict[int, Vehicle] = {}
        self.max_age = max_age
        self.next_id = 1
        self.removed_vehicles = []

        # Калибровочные параметры
        self.calibration_data = {
            'pixel_to_meter': 0.05,  # начальное приближение
            'reference_speeds': [],  # для калибровки
            'calibrated': False
        }

    def update(self, detections, current_time, frame_shape):
        """Update vehicle tracks with improved speed calculation"""
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

                # Расчет пройденного расстояния
                if vehicle.positions:
                    last_x, last_y = vehicle.positions[-1]
                    pixel_distance = math.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)
                    vehicle.distance_traveled += pixel_distance * self.calibration_data['pixel_to_meter']

                vehicle.positions.append((cx, cy))
                vehicle.timestamps.append(current_time)
                vehicle.confidence_history.append(confidence)
                vehicle.bbox_history.append(box)
                vehicle.active = True
                vehicle.last_detection_time = current_time
                vehicle.track_fail_count = 0

                # Калибровка для новых транспортных средств
                if not vehicle.calibrated and vehicle.calibration_frames < 10:
                    vehicle.calibration_frames += 1
                    if vehicle.calibration_frames >= 5:
                        vehicle.calibrated = True

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

                class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
                new_vehicle.vehicle_class = class_names.get(class_id, "car")

                self.vehicles[vehicle_id] = new_vehicle
                used_ids.add(vehicle_id)

        # Расчет скорости для всех активных транспортных средств
        for vehicle in self.vehicles.values():
            if vehicle.active and len(vehicle.positions) >= 2:
                current_speed = self._calculate_speed(vehicle, current_time)
                if current_speed is not None:
                    vehicle.speeds.append(current_speed)

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

    def _calculate_speed(self, vehicle: Vehicle, current_time: float) -> Optional[float]:
        """Точный расчет скорости в км/ч"""
        if len(vehicle.positions) < 2:
            return None

        # Используем несколько последних позиций для сглаживания
        num_positions = min(5, len(vehicle.positions))
        positions = list(vehicle.positions)[-num_positions:]
        timestamps = list(vehicle.timestamps)[-num_positions:]

        total_pixel_distance = 0
        total_time = 0

        for i in range(1, len(positions)):
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]

            pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            time_diff = timestamps[i] - timestamps[i - 1]

            if time_diff > 0:
                total_pixel_distance += pixel_distance
                total_time += time_diff

        if total_time <= 0:
            return None

        # Средняя скорость в пикселях в секунду
        avg_pixel_speed = total_pixel_distance / total_time

        # Конвертация в км/ч с калибровочным коэффициентом
        speed_kmh = avg_pixel_speed * self.calibration_data['pixel_to_meter'] * 3.6

        # Фильтрация реалистичных скоростей (исключаем аномалии)
        if 0 <= speed_kmh <= 200:  # Реалистичный диапазон скоростей
            return speed_kmh
        else:
            return None

    def auto_calibrate(self, current_time: float, reference_speed_kmh: float = 90.0):
        """Автоматическая калибровка на основе эталонной скорости"""
        active_vehicles = self.get_active_vehicles()
        if not active_vehicles:
            return

        # Берем транспортные средства с надежным трекингом
        reliable_vehicles = []
        for vehicle in active_vehicles:
            if (len(vehicle.speeds) >= 3 and
                    vehicle.calibrated and
                    current_time - vehicle.first_detection_time > 2.0):
                reliable_vehicles.append(vehicle)

        if not reliable_vehicles:
            return

        # Рассчитываем среднюю скорость в пикселях/сек
        total_pixel_speed = 0
        count = 0

        for vehicle in reliable_vehicles:
            if vehicle.speeds:
                # Преобразуем deque в list для среза
                speeds_list = list(vehicle.speeds)
                if len(speeds_list) >= 3:
                    current_speed_kmh = np.mean(speeds_list[-3:])
                    if 50 <= current_speed_kmh <= 120:
                        # Обратный расчет пиксельной скорости
                        pixel_speed = current_speed_kmh / (self.calibration_data['pixel_to_meter'] * 3.6)
                        total_pixel_speed += pixel_speed
                        count += 1

        if count > 0:
            avg_pixel_speed = total_pixel_speed / count
            # Обновляем коэффициент калибровки
            new_pixel_to_meter = reference_speed_kmh / (avg_pixel_speed * 3.6)

            # Плавное обновление коэффициента
            self.calibration_data['pixel_to_meter'] = (
                    0.7 * self.calibration_data['pixel_to_meter'] +
                    0.3 * new_pixel_to_meter
            )
            self.calibration_data['calibrated'] = True

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

    def is_speeding(self, speed: float, vehicle_class: str) -> bool:
        """Проверка превышения скорости"""
        speed_limits = {
            "car": 120,
            "motorcycle": 120,
            "bus": 90,
            "truck": 90
        }
        limit = speed_limits.get(vehicle_class, 120)
        return speed > limit

    def is_too_slow(self, speed: float, vehicle_class: str) -> bool:
        """Проверка слишком медленной скорости"""
        min_speeds = {
            "car": 60,
            "motorcycle": 60,
            "bus": 50,
            "truck": 50
        }
        min_speed = min_speeds.get(vehicle_class, 60)
        return speed < min_speed


class FrameProcessor:
    """Improved frame processor with accurate speed detection"""

    def __init__(self, model, wiedemann_model, tracker):
        self.model = model
        self.wiedemann = wiedemann_model
        self.tracker = tracker
        self.frame_queue = queue.Queue(maxsize=20)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.last_calibration_time = 0
        self.calibration_interval = 5.0  # калибровка каждые 5 секунд

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
        """Process single frame with accurate speed detection"""
        h, w = frame.shape[:2]

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

        used_ids = self.tracker.update(detections, current_time, frame.shape)

        # Периодическая калибровка
        if current_time - self.last_calibration_time > self.calibration_interval:
            self.tracker.auto_calibrate(current_time, 90.0)  # калибровка на 90 км/ч
            self.last_calibration_time = current_time

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

            # Расчет ускорения
            acceleration = self._calculate_acceleration(vehicle)
            if acceleration is not None:
                vehicle.accelerations.append(acceleration)

            # Текущая скорость
            current_speed = list(vehicle.speeds)[-1] if vehicle.speeds else 0

            violations = self._detect_violations(vehicle, current_time, current_speed)

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
                'speed': current_speed,
                'violations': sustained_violations,
                'is_aggressive': vehicle.is_aggressive,
                'active': vehicle.active,
                'vehicle_class': vehicle.vehicle_class,
                'distance_traveled': vehicle.distance_traveled
            }

            vehicles_data.append(vehicle_data)

        return {
            'frame': frame.copy(),
            'vehicles_data': vehicles_data,
            'frame_count': frame_count
        }

    def _calculate_acceleration(self, vehicle: Vehicle) -> Optional[float]:
        """Calculate acceleration in m/s² with smoothing"""
        if len(vehicle.speeds) < 3:
            return 0

        speeds = list(vehicle.speeds)[-3:]
        timestamps = list(vehicle.timestamps)[-3:]

        if len(speeds) < 2:
            return 0

        accelerations = []
        for i in range(1, len(speeds)):
            time_diff = timestamps[i] - timestamps[i - 1]
            if time_diff > 0:
                # Конвертация скорости из км/ч в м/с
                speed_diff_ms = (speeds[i] - speeds[i - 1]) / 3.6
                accel = speed_diff_ms / time_diff
                accelerations.append(accel)

        if not accelerations:
            return 0

        return sum(accelerations) / len(accelerations)

    def _detect_violations(self, vehicle: Vehicle, current_time: float, current_speed: float) -> List[str]:
        violations = []

        if current_time - vehicle.first_detection_time < 2.0:
            return violations

        if len(vehicle.positions) < 8:
            return violations

        # Проверка скорости
        if self.wiedemann.is_speeding(current_speed, vehicle.vehicle_class):
            violations.append('speeding')

        if self.wiedemann.is_too_slow(current_speed, vehicle.vehicle_class):
            violations.append('too_slow')

        # Проверка ускорения/торможения
        current_accel = list(vehicle.accelerations)[-1] if vehicle.accelerations else 0

        if (current_accel > 0 and
                self.wiedemann.is_aggressive_acceleration(current_accel, vehicle.vehicle_class)):
            violations.append('rapid_acceleration')

        if (current_accel < 0 and
                self.wiedemann.is_aggressive_deceleration(current_accel, vehicle.vehicle_class)):
            violations.append('rapid_braking')

        return violations


class TrafficAnalyzer:
    """Final improved traffic analyzer with accurate speed detection"""

    VIOLATION_CATEGORIES = {
        'speeding': 'Speeding',
        'too_slow': 'Too Slow',
        'rapid_acceleration': 'Rapid Acceleration',
        'rapid_braking': 'Rapid Braking'
    }

    def __init__(self, video_path: str, model_path: str = 'yolo11x.pt'):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.wiedemann = WiedemannModel()
        self.tracker = ImprovedVehicleTracker(max_age=4.0)

        self.violation_stats = defaultdict(int)
        self.aggressive_drivers = set()
        self.speed_stats = []

        self.fps = 0
        self.frame_count = 0

        self.frame_processor = FrameProcessor(self.model, self.wiedemann, self.tracker)
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

                    # Статистика скоростей
                    speeds = []
                    for vehicle in self.tracker.get_active_vehicles():
                        if vehicle.speeds:
                            # Преобразуем deque в list для среза
                            speeds_list = list(vehicle.speeds)
                            if len(speeds_list) >= 3:
                                speeds.append(np.mean(speeds_list[-3:]))

                    avg_speed = np.mean(speeds) if speeds else 0

                    print(
                        f"Progress: {progress:.1f}% | Active: {active_vehicles} | Total: {all_vehicles} | Aggressive: {aggressive} | Avg Speed: {avg_speed:.1f} km/h")
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

        for vehicle in vehicles_data:
            x1, y1, x2, y2 = vehicle['box']
            cx, cy = vehicle['center']
            violations = vehicle['violations']
            is_aggressive = vehicle['is_aggressive']
            vehicle_class = vehicle.get('vehicle_class', 'car')
            speed = vehicle.get('speed', 0)

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
            info_text = f"ID:{vehicle['id']}({class_symbol}) {speed:.1f}km/h"
            cv2.putText(frame, info_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Отображение нарушений
            for i, violation in enumerate(violations[:2]):
                viol_text = self.VIOLATION_CATEGORIES.get(violation, violation)
                cv2.putText(frame, viol_text, (x1, y2 + 15 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Статистика
        stats_bg = np.zeros((140, 500, 3), dtype=np.uint8)
        frame[10:150, 10:510] = stats_bg

        active_vehicles = len(vehicles_data)
        total_tracked = len(self.tracker.get_all_analyzed_vehicles())
        aggressive_count = len(self.aggressive_drivers)

        # Расчет средней скорости
        speeds = []
        for vehicle in self.tracker.get_active_vehicles():
            if vehicle.speeds:
                # Преобразуем deque в list для среза
                speeds_list = list(vehicle.speeds)
                if len(speeds_list) >= 3:
                    speeds.append(np.mean(speeds_list[-3:]))
        avg_speed = np.mean(speeds) if speeds else 0

        cv2.putText(frame, f"Active vehicles: {active_vehicles}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total tracked: {total_tracked}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Aggressive: {aggressive_count}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if aggressive_count > 0 else (255, 255, 255), 2)
        cv2.putText(frame, f"Avg speed: {avg_speed:.1f} km/h", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Calibration: {'ON' if self.tracker.calibration_data['calibrated'] else 'OFF'}",
                    (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def _update_statistics(self, processed_data: Dict):
        """Update statistics"""
        for vehicle_data in processed_data['vehicles_data']:
            vehicle_id = vehicle_data['id']
            speed = vehicle_data.get('speed', 0)

            if speed > 0:
                self.speed_stats.append(speed)

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

        for vehicle in all_vehicles:
            vehicle_types[vehicle.vehicle_class] += 1

        violation_percentages = {}
        for viol_key, viol_name in self.VIOLATION_CATEGORIES.items():
            count = self.violation_stats.get(viol_key, 0)
            pct = (count / max(total_tracked, 1)) * 100
            violation_percentages[viol_name] = {
                'count': count,
                'percentage': round(pct, 2)
            }

        # Статистика скоростей
        speed_stats = {}
        if self.speed_stats:
            speed_stats = {
                'average_speed': round(np.mean(self.speed_stats), 2),
                'min_speed': round(np.min(self.speed_stats), 2),
                'max_speed': round(np.max(self.speed_stats), 2),
                'speed_std': round(np.std(self.speed_stats), 2),
                'speed_distribution': {
                    '70-80 km/h': len([s for s in self.speed_stats if 70 <= s < 80]),
                    '80-90 km/h': len([s for s in self.speed_stats if 80 <= s < 90]),
                    '90-100 km/h': len([s for s in self.speed_stats if 90 <= s < 100]),
                    '100-110 km/h': len([s for s in self.speed_stats if 100 <= s < 110]),
                    '110-120 km/h': len([s for s in self.speed_stats if 110 <= s < 120]),
                    '120+ km/h': len([s for s in self.speed_stats if s >= 120])
                }
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
            'speed_statistics': speed_stats,
            'vehicle_type_distribution': dict(vehicle_types),
            'violations_breakdown': violation_percentages,
            'analysis_parameters': {
                'detection_confidence': 0.6,
                'tracking_max_age_seconds': 4.0,
                'speed_calibration': 'auto',
                'calibration_reference_speed': '90 km/h'
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

        print(f"\nSPEED STATISTICS:")
        print("-" * 50)
        if speed_stats:
            print(f"Average speed: {speed_stats['average_speed']:.1f} km/h")
            print(f"Speed range: {speed_stats['min_speed']:.1f} - {speed_stats['max_speed']:.1f} km/h")
            print(f"Speed distribution:")
            for range_name, count in speed_stats['speed_distribution'].items():
                if count > 0:
                    pct = (count / len(self.speed_stats)) * 100
                    print(f"  {range_name}: {count} vehicles ({pct:.1f}%)")

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

    try:
        print("Starting traffic analyzer with accurate speed detection...")
        print("Initializing systems...")

        analyzer = TrafficAnalyzer(VIDEO_PATH)
        analyzer.process_video(OUTPUT_VIDEO, STATS_JSON)

    except FileNotFoundError:
        print("Error: Video file not found! Please check the path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()