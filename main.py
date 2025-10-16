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
from scipy import stats
from scipy.spatial import distance as dist
import warnings

warnings.filterwarnings('ignore')


@dataclass
class Vehicle:
    id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=50))
    speeds: deque = field(default_factory=lambda: deque(maxlen=20))
    accelerations: deque = field(default_factory=lambda: deque(maxlen=20))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))
    violation_history: Dict[str, int] = field(default_factory=dict)
    is_aggressive: bool = False
    vehicle_class: str = "car"
    first_detection_time: float = 0
    last_detection_time: float = 0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=15))
    track_fail_count: int = 0
    active: bool = True
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=15))
    distance_traveled: float = 0
    last_speed_calc_time: float = 0
    calibrated: bool = False
    calibration_frames: int = 0
    speed_confidence: float = 0.0
    trajectory_angle: float = 0.0
    stability_score: float = 0.0
    last_reliable_speed: float = 0.0
    speed_std: float = 0.0


class KalmanFilter:
    """Простой фильтр Калмана для сглаживания скорости"""

    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimated_speed = 0.0
        self.estimation_error = 1.0

    def update(self, measurement):
        # Prediction
        pred_error = self.estimation_error + self.process_noise

        # Update
        kalman_gain = pred_error / (pred_error + self.measurement_noise)
        self.estimated_speed = self.estimated_speed + kalman_gain * (measurement - self.estimated_speed)
        self.estimation_error = (1 - kalman_gain) * pred_error

        return self.estimated_speed


class ImprovedVehicleTracker:

    def __init__(self, max_age=3.0):
        self.vehicles: Dict[int, Vehicle] = {}
        self.max_age = max_age
        self.next_id = 1
        self.removed_vehicles = []

        # Улучшенные калибровочные параметры
        self.calibration_data = {
            'pixel_to_meter': 0.05,
            'reference_speeds': [],
            'calibrated': False,
            'calibration_samples': 0,
            'calibration_confidence': 0.0
        }

        # Фильтры Калмана для каждого транспортного средства
        self.kalman_filters = {}

        # Статистика для анализа качества трекинга
        self.tracking_stats = {
            'total_matches': 0,
            'total_detections': 0,
            'match_ratio': 0.0
        }

    def update(self, detections, current_time, frame_shape):
        """Улучшенное обновление треков с продвинутым сопоставлением"""
        h, w = frame_shape[:2]

        for vehicle in self.vehicles.values():
            vehicle.active = False

        used_ids = set()

        # Статистика детекций
        self.tracking_stats['total_detections'] += len(detections)

        # Первый проход: сопоставление существующих треков
        unmatched_detections = list(range(len(detections)))
        matched_pairs = []

        for i, detection in enumerate(detections):
            box, confidence, class_id = detection
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            best_match_id = None
            best_score = 0
            max_match_distance = 200  # Увеличенный радиус сопоставления

            for vehicle_id, vehicle in self.vehicles.items():
                if not vehicle.positions or vehicle_id in used_ids:
                    continue

                last_x, last_y = vehicle.positions[-1]
                pixel_distance = math.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)

                if pixel_distance < max_match_distance:
                    # Многофакторная оценка схожести
                    score = self._calculate_similarity_score(vehicle, detection, pixel_distance, current_time)

                    if score > best_score and score > 0.4:  # Повышенный порог
                        best_score = score
                        best_match_id = vehicle_id

            if best_match_id is not None:
                matched_pairs.append((best_match_id, i))
                unmatched_detections.remove(i)
                used_ids.add(best_match_id)
                self.tracking_stats['total_matches'] += 1

        # Обновление сопоставленных треков
        for vehicle_id, det_idx in matched_pairs:
            vehicle = self.vehicles[vehicle_id]
            box, confidence, class_id = detections[det_idx]
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

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

            if not vehicle.calibrated and vehicle.calibration_frames < 15:
                vehicle.calibration_frames += 1
                if vehicle.calibration_frames >= 8:
                    vehicle.calibrated = True

        # Создание новых треков для несопоставленных детекций
        for det_idx in unmatched_detections:
            box, confidence, class_id = detections[det_idx]
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area < 800:  # Фильтрация мелких объектов
                continue

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

            # Инициализация фильтра Калмана
            self.kalman_filters[vehicle_id] = KalmanFilter()

            self.vehicles[vehicle_id] = new_vehicle
            used_ids.add(vehicle_id)

        # Расчет скорости и улучшенной статистики
        for vehicle in self.vehicles.values():
            if vehicle.active and len(vehicle.positions) >= 3:
                current_speed = self._calculate_improved_speed(vehicle, current_time)
                if current_speed is not None:
                    # Применение фильтра Калмана
                    if vehicle.id in self.kalman_filters:
                        filtered_speed = self.kalman_filters[vehicle.id].update(current_speed)
                        vehicle.speeds.append(filtered_speed)
                        vehicle.last_reliable_speed = filtered_speed
                    else:
                        vehicle.speeds.append(current_speed)
                        vehicle.last_reliable_speed = current_speed

                    # Расчет стабильности скорости
                    if len(vehicle.speeds) >= 5:
                        speed_array = np.array(list(vehicle.speeds)[-5:])
                        vehicle.speed_std = np.std(speed_array)
                        vehicle.speed_confidence = max(0, 1 - (vehicle.speed_std / 20))  # 20 км/ч как базовый шум

                    # Расчет угла траектории
                    if len(vehicle.positions) >= 5:
                        vehicle.trajectory_angle = self._calculate_trajectory_angle(vehicle)

        # Управление жизненным циклом треков
        to_remove = []
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.active:
                vehicle.track_fail_count += 1

            removal_condition = (
                    current_time - vehicle.last_detection_time > self.max_age or
                    vehicle.track_fail_count > 60 or
                    (len(vehicle.positions) > 10 and vehicle.speed_confidence < 0.3)
            )

            if removal_condition:
                if (vehicle.last_detection_time - vehicle.first_detection_time) > 3.0:
                    self.removed_vehicles.append(vehicle)
                to_remove.append(vehicle_id)

                # Удаление фильтра Калмана
                if vehicle_id in self.kalman_filters:
                    del self.kalman_filters[vehicle_id]

        for vehicle_id in to_remove:
            del self.vehicles[vehicle_id]

        # Обновление статистики сопоставления
        if self.tracking_stats['total_detections'] > 0:
            self.tracking_stats['match_ratio'] = (
                    self.tracking_stats['total_matches'] / self.tracking_stats['total_detections']
            )

        return used_ids

    def _calculate_similarity_score(self, vehicle: Vehicle, detection, pixel_distance: float,
                                    current_time: float) -> float:
        """Многофакторная оценка схожести для улучшенного сопоставления"""
        box, confidence, class_id = detection
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        last_x, last_y = vehicle.positions[-1]

        # 1. Расстояние (40% веса)
        max_dist = 200
        distance_score = 1 - (pixel_distance / max_dist)

        # 2. Размер (25% веса)
        if vehicle.bbox_history:
            last_bbox = vehicle.bbox_history[-1]
            last_area = (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1])
            current_area = (x2 - x1) * (y2 - y1)
            size_similarity = 1 - abs(current_area - last_area) / max(current_area, last_area)
        else:
            size_similarity = 0.5

        # 3. Направление движения (20% веса)
        direction_score = 1.0
        if len(vehicle.positions) >= 3:
            # Предсказание следующей позиции
            prev_x, prev_y = vehicle.positions[-2]
            dx = last_x - prev_x
            dy = last_y - prev_y

            predicted_x = last_x + dx
            predicted_y = last_y + dy

            prediction_error = math.sqrt((cx - predicted_x) ** 2 + (cy - predicted_y) ** 2)
            direction_score = 1 - min(1.0, prediction_error / 100)

        # 4. Временная согласованность (15% веса)
        time_diff = current_time - vehicle.last_detection_time
        time_score = 1 - min(1.0, time_diff / 2.0)  # Штраф за большие пропуски

        total_score = (
                distance_score * 0.4 +
                size_similarity * 0.25 +
                direction_score * 0.2 +
                time_score * 0.15
        )

        return total_score

    def _calculate_improved_speed(self, vehicle: Vehicle, current_time: float) -> Optional[float]:
        """Улучшенный расчет скорости с использованием регрессии"""
        if len(vehicle.positions) < 3:
            return None

        # Используем больше точек для лучшей точности
        num_positions = min(10, len(vehicle.positions))
        positions = list(vehicle.positions)[-num_positions:]
        timestamps = list(vehicle.timestamps)[-num_positions:]

        # Линейная регрессия для нахождения скорости
        try:
            # Преобразование в numpy массивы
            times = np.array(timestamps)
            x_coords = np.array([p[0] for p in positions])
            y_coords = np.array([p[1] for p in positions])

            # Регрессия для X и Y координат
            slope_x, _, r_value_x, _, _ = stats.linregress(times, x_coords)
            slope_y, _, r_value_y, _, _ = stats.linregress(times, y_coords)

            # Качество регрессии
            regression_quality = (abs(r_value_x) + abs(r_value_y)) / 2

            if regression_quality < 0.7:  # Низкое качество регрессии
                return self._calculate_simple_speed(vehicle, current_time)

            # Скорость в пикселях в секунду
            pixel_speed = math.sqrt(slope_x ** 2 + slope_y ** 2)

            # Конвертация в км/ч
            speed_kmh = pixel_speed * self.calibration_data['pixel_to_meter'] * 3.6

            # Взвешивание по качеству регрессии
            weighted_speed = speed_kmh * regression_quality

            if 0 <= weighted_speed <= 200:
                vehicle.stability_score = regression_quality
                return weighted_speed
            else:
                return self._calculate_simple_speed(vehicle, current_time)

        except:
            # Fallback на простой метод
            return self._calculate_simple_speed(vehicle, current_time)

    def _calculate_simple_speed(self, vehicle: Vehicle, current_time: float) -> Optional[float]:
        """Простой расчет скорости как fallback"""
        if len(vehicle.positions) < 2:
            return None

        positions = list(vehicle.positions)[-5:]  # Используем 5 последних позиций
        timestamps = list(vehicle.timestamps)[-5:]

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

        avg_pixel_speed = total_pixel_distance / total_time
        speed_kmh = avg_pixel_speed * self.calibration_data['pixel_to_meter'] * 3.6

        if 0 <= speed_kmh <= 200:
            return speed_kmh
        return None

    def _calculate_trajectory_angle(self, vehicle: Vehicle) -> float:
        """Расчет угла траектории движения"""
        if len(vehicle.positions) < 5:
            return 0.0

        positions = list(vehicle.positions)
        # Берем первую и последнюю точку из последних 5 позиций
        start_idx = max(0, len(positions) - 5)
        x1, y1 = positions[start_idx]
        x2, y2 = positions[-1]

        if x2 == x1:  # Избегаем деления на ноль
            return 90.0 if y2 > y1 else -90.0

        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def auto_calibrate(self, current_time: float, reference_speed_kmh: float = 90.0):
        """Улучшенная автоматическая калибровка"""
        active_vehicles = self.get_active_vehicles()
        if not active_vehicles:
            return

        # Отбираем только надежные треки
        reliable_vehicles = []
        for vehicle in active_vehicles:
            if (len(vehicle.speeds) >= 5 and
                    vehicle.calibrated and
                    current_time - vehicle.first_detection_time > 3.0 and
                    vehicle.speed_confidence > 0.6):
                reliable_vehicles.append(vehicle)

        if len(reliable_vehicles) < 3:  # Нужно минимум 3 надежных трека
            return

        # Собираем скорости для калибровки
        calibration_speeds = []
        for vehicle in reliable_vehicles:
            speeds_list = list(vehicle.speeds)
            if len(speeds_list) >= 5:
                recent_speeds = speeds_list[-5:]
                avg_speed = np.mean(recent_speeds)
                if 60 <= avg_speed <= 130:  # Реалистичный диапазон для калибровки
                    calibration_speeds.append(avg_speed)

        if len(calibration_speeds) < 3:
            return

        # Используем медиану для устойчивости к выбросам
        median_speed = np.median(calibration_speeds)

        # Рассчитываем новый коэффициент калибровки
        current_pixel_speed = median_speed / (self.calibration_data['pixel_to_meter'] * 3.6)
        new_pixel_to_meter = reference_speed_kmh / (current_pixel_speed * 3.6)

        # Плавное обновление с адаптивным learning rate
        learning_rate = max(0.1, min(0.5, 1.0 / (self.calibration_data['calibration_samples'] + 1)))

        self.calibration_data['pixel_to_meter'] = (
                (1 - learning_rate) * self.calibration_data['pixel_to_meter'] +
                learning_rate * new_pixel_to_meter
        )

        self.calibration_data['calibration_samples'] += 1
        self.calibration_data['calibrated'] = True

        # Расчет уверенности в калибровке
        speed_variance = np.var(calibration_speeds)
        self.calibration_data['calibration_confidence'] = max(0, 1 - (speed_variance / 400))  # 400 = 20^2

    def get_active_vehicles(self):
        return [v for v in self.vehicles.values() if v.active]

    def get_all_analyzed_vehicles(self):
        return list(self.vehicles.values()) + self.removed_vehicles

    def get_tracking_quality(self) -> float:
        """Возвращает качество трекинга (0-1)"""
        return self.tracking_stats['match_ratio']


class AdvancedWiedemannModel:

    def __init__(self):
        # Базовые параметры модели Видемана
        self.cc0 = 2.0
        self.cc1 = 1.2
        self.cc2 = 4.0
        self.cc4 = -0.35
        self.cc5 = 0.25
        self.cc8 = 2.0
        self.cc9 = 3.0

        # Адаптивные пороги для разных типов транспортных средств
        self.adaptive_thresholds = {
            "car": {
                "max_speed": 120,
                "min_speed": 60,
                "accel_threshold": 3.0,
                "decel_threshold": -3.5,
                "speed_variance_threshold": 15.0
            },
            "motorcycle": {
                "max_speed": 120,
                "min_speed": 60,
                "accel_threshold": 4.0,
                "decel_threshold": -4.5,
                "speed_variance_threshold": 20.0
            },
            "bus": {
                "max_speed": 90,
                "min_speed": 50,
                "accel_threshold": 2.0,
                "decel_threshold": -2.5,
                "speed_variance_threshold": 10.0
            },
            "truck": {
                "max_speed": 90,
                "min_speed": 50,
                "accel_threshold": 1.5,
                "decel_threshold": -2.0,
                "speed_variance_threshold": 8.0
            }
        }

    def is_aggressive_acceleration(self, acceleration: float, vehicle_class: str) -> bool:
        thresholds = self.adaptive_thresholds.get(vehicle_class, self.adaptive_thresholds["car"])
        return acceleration > thresholds["accel_threshold"]

    def is_aggressive_deceleration(self, deceleration: float, vehicle_class: str) -> bool:
        thresholds = self.adaptive_thresholds.get(vehicle_class, self.adaptive_thresholds["car"])
        return deceleration < thresholds["decel_threshold"]

    def is_speeding(self, speed: float, vehicle_class: str) -> bool:
        thresholds = self.adaptive_thresholds.get(vehicle_class, self.adaptive_thresholds["car"])
        return speed > thresholds["max_speed"]

    def is_too_slow(self, speed: float, vehicle_class: str) -> bool:
        thresholds = self.adaptive_thresholds.get(vehicle_class, self.adaptive_thresholds["car"])
        return speed < thresholds["min_speed"]

    def has_unsafe_speed_variance(self, speed_std: float, vehicle_class: str) -> bool:
        """Обнаружение нестабильной скорости"""
        thresholds = self.adaptive_thresholds.get(vehicle_class, self.adaptive_thresholds["car"])
        return speed_std > thresholds["speed_variance_threshold"]

    def calculate_aggression_score(self, violations: List[str], speed_std: float,
                                   current_speed: float, vehicle_class: str) -> float:
        """Расчет агрессивности вождения (0-100)"""
        score = 0.0

        # Штрафы за нарушения
        violation_weights = {
            'speeding': 30,
            'too_slow': 10,
            'rapid_acceleration': 20,
            'rapid_braking': 20,
            'unsafe_speed_variance': 15
        }

        for violation in violations:
            score += violation_weights.get(violation, 0)

        # Дополнительные факторы
        if self.is_speeding(current_speed, vehicle_class):
            speed_excess = current_speed - self.adaptive_thresholds[vehicle_class]["max_speed"]
            score += min(20, speed_excess / 2)

        if self.has_unsafe_speed_variance(speed_std, vehicle_class):
            score += min(15, speed_std / 2)

        return min(100, score)


class AdvancedFrameProcessor:
    """Продвинутый обработчик кадров с улучшенной детекцией скорости"""

    def __init__(self, model, wiedemann_model, tracker):
        self.model = model
        self.wiedemann = wiedemann_model
        self.tracker = tracker
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.last_calibration_time = 0
        self.calibration_interval = 3.0  # Более частая калибровка

        # Статистика обработки
        self.processing_stats = {
            'frames_processed': 0,
            'avg_processing_time': 0,
            'last_processing_time': 0
        }

    def process_frames(self):
        """Улучшенная обработка кадров в отдельном потоке"""
        while not self.stop_event.is_set():
            try:
                start_time = time.time()
                frame_data = self.frame_queue.get(timeout=1)
                if frame_data is None:
                    break

                frame, frame_count, current_time = frame_data
                processed_data = self._process_single_frame(frame, frame_count, current_time)
                self.result_queue.put(processed_data)

                # Обновление статистики обработки
                processing_time = time.time() - start_time
                self._update_processing_stats(processing_time)

            except queue.Empty:
                continue

    def _update_processing_stats(self, processing_time: float):
        """Обновление статистики времени обработки"""
        self.processing_stats['frames_processed'] += 1
        self.processing_stats['last_processing_time'] = processing_time

        # Экспоненциальное скользящее среднее
        alpha = 0.1
        if self.processing_stats['avg_processing_time'] == 0:
            self.processing_stats['avg_processing_time'] = processing_time
        else:
            self.processing_stats['avg_processing_time'] = (
                    alpha * processing_time +
                    (1 - alpha) * self.processing_stats['avg_processing_time']
            )

    def _process_single_frame(self, frame: np.ndarray, frame_count: int, current_time: float):
        """Улучшенная обработка одиночного кадра"""
        h, w = frame.shape[:2]

        # Детекция транспортных средств с оптимизированными параметрами
        results = self.model(frame, classes=[2, 3, 5, 7],
                             verbose=False, conf=0.55, iou=0.6)  # Оптимизированные пороги

        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if conf >= 0.55:  # Пониженный порог для лучшего покрытия
                    detections.append((box, conf, class_id))

        used_ids = self.tracker.update(detections, current_time, frame.shape)

        # Адаптивная калибровка
        if current_time - self.last_calibration_time > self.calibration_interval:
            active_vehicles = self.tracker.get_active_vehicles()
            if len(active_vehicles) >= 2:  # Калибруем только если есть достаточное количество транспортных средств
                self.tracker.auto_calibrate(current_time, 85.0)  # Адаптивная эталонная скорость
            self.last_calibration_time = current_time

        vehicles_data = []
        for vehicle in self.tracker.get_active_vehicles():
            if not vehicle.positions or len(vehicle.positions) < 3:
                continue

            cx, cy = vehicle.positions[-1]

            # Поиск соответствующего bounding box
            vehicle_box = self._find_best_matching_bbox(vehicle, detections, cx, cy)
            if vehicle_box is None and vehicle.bbox_history:
                vehicle_box = vehicle.bbox_history[-1]

            if vehicle_box is None:
                continue

            # Расчет улучшенного ускорения
            acceleration = self._calculate_smoothed_acceleration(vehicle)
            if acceleration is not None:
                vehicle.accelerations.append(acceleration)

            # Используем последнюю надежную скорость
            current_speed = vehicle.last_reliable_speed

            # Детекция нарушений с улучшенной логикой
            violations = self._detect_advanced_violations(vehicle, current_time, current_speed)

            # Обновление истории нарушений
            current_violations = set()
            for violation in violations:
                vehicle.violation_history[violation] = vehicle.violation_history.get(violation, 0) + 1
                current_violations.add(violation)

            # Определение устойчивых нарушений
            sustained_violations = []
            for violation, count in vehicle.violation_history.items():
                if count >= 2:  # Пониженный порог для быстрого реагирования
                    sustained_violations.append(violation)

            # Расчет агрессивности
            aggression_score = self.wiedemann.calculate_aggression_score(
                sustained_violations, vehicle.speed_std, current_speed, vehicle.vehicle_class
            )
            vehicle.is_aggressive = aggression_score > 40  # Порог агрессивности

            vehicle_data = {
                'id': vehicle.id,
                'box': [int(x) for x in vehicle_box],
                'center': (cx, cy),
                'speed': current_speed,
                'speed_confidence': vehicle.speed_confidence,
                'violations': sustained_violations,
                'is_aggressive': vehicle.is_aggressive,
                'aggression_score': aggression_score,
                'active': vehicle.active,
                'vehicle_class': vehicle.vehicle_class,
                'distance_traveled': vehicle.distance_traveled,
                'trajectory_angle': vehicle.trajectory_angle,
                'stability_score': vehicle.stability_score
            }

            vehicles_data.append(vehicle_data)

        return {
            'frame': frame.copy(),
            'vehicles_data': vehicles_data,
            'frame_count': frame_count,
            'processing_stats': self.processing_stats.copy(),
            'tracking_quality': self.tracker.get_tracking_quality(),
            'calibration_confidence': self.tracker.calibration_data['calibration_confidence']
        }

    def _find_best_matching_bbox(self, vehicle: Vehicle, detections: List, cx: int, cy: int):
        """Поиск наилучшего соответствия bounding box"""
        best_box = None
        best_distance = float('inf')

        for box, conf, class_id in detections:
            box_cx = int((box[0] + box[2]) / 2)
            box_cy = int((box[1] + box[3]) / 2)
            distance = math.sqrt((cx - box_cx) ** 2 + (cy - box_cy) ** 2)

            if distance < best_distance and distance < 100:  # Увеличенный радиус поиска
                best_distance = distance
                best_box = box

        return best_box

    def _calculate_smoothed_acceleration(self, vehicle: Vehicle) -> Optional[float]:
        """Сглаженный расчет ускорения"""
        if len(vehicle.speeds) < 4:
            return 0

        # Используем сглаженные скорости
        speeds = list(vehicle.speeds)[-6:]  # Больше точек для сглаживания
        timestamps = list(vehicle.timestamps)[-6:]

        if len(speeds) < 3:
            return 0

        # Расчет ускорения через линейную регрессию
        try:
            times = np.array(timestamps[-4:])  # Используем последние 4 точки
            speed_values = np.array(speeds[-4:])

            # Линейная регрессия скорости от времени
            slope, _, r_value, _, _ = stats.linregress(times, speed_values)

            # Конвертация в м/с²
            acceleration = (slope / 3.6)  # км/ч/с -> м/с²

            # Учитываем качество регрессии
            if abs(r_value) > 0.6:
                return acceleration
            else:
                return self._calculate_simple_acceleration(vehicle)

        except:
            return self._calculate_simple_acceleration(vehicle)

    def _calculate_simple_acceleration(self, vehicle: Vehicle) -> float:
        """Простой расчет ускорения"""
        if len(vehicle.speeds) < 3:
            return 0

        speeds = list(vehicle.speeds)[-3:]
        timestamps = list(vehicle.timestamps)[-3:]

        accelerations = []
        for i in range(1, len(speeds)):
            time_diff = timestamps[i] - timestamps[i - 1]
            if time_diff > 0:
                speed_diff_ms = (speeds[i] - speeds[i - 1]) / 3.6
                accel = speed_diff_ms / time_diff
                accelerations.append(accel)

        return np.mean(accelerations) if accelerations else 0

    def _detect_advanced_violations(self, vehicle: Vehicle, current_time: float, current_speed: float) -> List[str]:
        """Улучшенная детекция нарушений"""
        violations = []

        if current_time - vehicle.first_detection_time < 2.5:
            return violations

        if len(vehicle.positions) < 10:
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

        if self.wiedemann.has_unsafe_speed_variance(vehicle.speed_std, vehicle.vehicle_class):
            violations.append('unsafe_speed_variance')

        return violations


class AdvancedTrafficAnalyzer:
    """Продвинутый анализатор трафика с улучшенными функциями"""

    VIOLATION_CATEGORIES = {
        'speeding': 'Speeding',
        'too_slow': 'Too Slow',
        'rapid_acceleration': 'Rapid Acceleration',
        'rapid_braking': 'Rapid Braking',
        'unsafe_speed_variance': 'Unstable Speed'
    }

    def __init__(self, video_path: str, model_path: str = 'yolo11x.pt'):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.wiedemann = AdvancedWiedemannModel()
        self.tracker = ImprovedVehicleTracker(max_age=5.0)  # Увеличенное время жизни

        self.violation_stats = defaultdict(int)
        self.aggressive_drivers = set()
        self.speed_stats = []
        self.analysis_start_time = time.time()

        self.fps = 0
        self.frame_count = 0
        self.real_fps = 0
        self.last_fps_update = time.time()

        self.frame_processor = AdvancedFrameProcessor(self.model, self.wiedemann, self.tracker)
        self.processor_thread = None

        # Расширенная статистика
        self.extended_stats = {
            'total_processing_time': 0,
            'peak_vehicles': 0,
            'speed_analysis': {},
            'time_distribution': defaultdict(int)
        }

    def process_video(self, output_path: str = 'output_analyzed.mp4',
                      stats_output: str = 'advanced_statistics.json'):
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

        print(f"🚀 Starting Advanced Traffic Analyzer")
        print(f"📹 Video: {self.video_path}")
        print(f"⚡ FPS: {self.fps:.1f}, Resolution: {w}x{h}, Frames: {total_frames}")
        print("=" * 70)

        self.frame_processor.stop_event.clear()
        self.processor_thread = threading.Thread(target=self.frame_processor.process_frames)
        self.processor_thread.start()

        last_stats_time = time.time()
        processed_frames = 0
        frame_times = deque(maxlen=30)

        try:
            while True:
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                current_time = self.frame_count / self.fps

                try:
                    self.frame_processor.frame_queue.put(
                        (frame, self.frame_count, current_time),
                        timeout=0.3
                    )
                except queue.Full:
                    continue

                try:
                    processed_data = self.frame_processor.result_queue.get(timeout=0.3)
                    frame = self._advanced_visualization(processed_data, w, h)
                    self._update_extended_statistics(processed_data)
                    processed_frames += 1
                except queue.Empty:
                    pass

                out.write(frame)

                # Расчет реального FPS
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                if len(frame_times) >= 10:
                    self.real_fps = 1.0 / (sum(frame_times) / len(frame_times))

                if time.time() - last_stats_time > 2.0:
                    self._print_live_stats(total_frames, current_time)
                    last_stats_time = time.time()

        except KeyboardInterrupt:
            print("\n🛑 Stopping analysis...")

        finally:
            self.frame_processor.stop_event.set()
            if self.processor_thread:
                self.processor_thread.join(timeout=5)

            cap.release()
            out.release()

            self.generate_advanced_statistics_report(stats_output)
            print(f"\n✅ Video saved: {output_path}")
            print(f"📊 Statistics saved: {stats_output}")

    def _print_live_stats(self, total_frames: int, current_time: float):
        """Вывод расширенной живой статистики"""
        progress = (self.frame_count / total_frames) * 100
        active_vehicles = len(self.tracker.get_active_vehicles())
        all_vehicles = len(self.tracker.get_all_analyzed_vehicles())
        aggressive = len(self.aggressive_drivers)

        # Статистика скоростей
        speeds = []
        for vehicle in self.tracker.get_active_vehicles():
            if vehicle.speeds and vehicle.speed_confidence > 0.5:
                speeds.append(vehicle.last_reliable_speed)

        avg_speed = np.mean(speeds) if speeds else 0
        speed_std = np.std(speeds) if speeds else 0

        # Качество системы
        tracking_quality = self.tracker.get_tracking_quality()
        calibration_conf = self.tracker.calibration_data['calibration_confidence']

        print(f"📊 Progress: {progress:5.1f}% | "
              f"🚗 Active: {active_vehicles:2d} | "
              f"📈 Total: {all_vehicles:3d} | "
              f"🔴 Aggressive: {aggressive:2d} | "
              f"⚡ Speed: {avg_speed:5.1f}±{speed_std:3.1f} km/h | "
              f"🎯 Track: {tracking_quality:.2f} | "
              f"🔧 Calib: {calibration_conf:.2f}")

    def _advanced_visualization(self, processed_data: Dict, w: int, h: int) -> np.ndarray:
        """Продвинутая визуализация с расширенной информацией"""
        frame = processed_data['frame']
        vehicles_data = processed_data['vehicles_data']
        tracking_quality = processed_data.get('tracking_quality', 0)
        calibration_conf = processed_data.get('calibration_confidence', 0)

        for vehicle in vehicles_data:
            frame = self._draw_advanced_vehicle_info(frame, vehicle)

        # Расширенная панель статистики
        frame = self._draw_advanced_stats_panel(frame, processed_data, w, h)

        return frame

    def _draw_advanced_vehicle_info(self, frame: np.ndarray, vehicle: Dict) -> np.ndarray:
        """Отрисовка расширенной информации о транспортном средстве"""
        x1, y1, x2, y2 = vehicle['box']
        cx, cy = vehicle['center']
        violations = vehicle['violations']
        is_aggressive = vehicle['is_aggressive']
        vehicle_class = vehicle.get('vehicle_class', 'car')
        speed = vehicle.get('speed', 0)
        speed_confidence = vehicle.get('speed_confidence', 0)
        aggression_score = vehicle.get('aggression_score', 0)

        # Определение цвета на основе агрессивности
        if is_aggressive:
            color = (0, 0, 255)  # Красный
        elif aggression_score > 20:
            color = (0, 165, 255)  # Оранжевый
        elif violations:
            color = (0, 255, 255)  # Желтый
        else:
            color = (0, 255, 0)  # Зеленый

        # Отрисовка bounding box
        thickness = 3 if is_aggressive else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(frame, (cx, cy), 6, color, -1)

        # Основная информация
        class_symbol = {"car": "🚗", "motorcycle": "🏍", "bus": "🚌", "truck": "🚛"}.get(vehicle_class, "🚙")
        info_text = f"ID:{vehicle['id']} {class_symbol} {speed:.0f}km/h"
        cv2.putText(frame, info_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Дополнительная информация
        if speed_confidence < 0.7:
            conf_text = f"Conf:{speed_confidence:.1f}"
            cv2.putText(frame, conf_text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Отображение нарушений
        for i, violation in enumerate(violations[:3]):
            viol_text = self.VIOLATION_CATEGORIES.get(violation, violation)
            cv2.putText(frame, viol_text, (x1, y2 + 15 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Индикатор агрессивности
        if aggression_score > 0:
            agg_text = f"Aggr:{aggression_score:.0f}%"
            cv2.putText(frame, agg_text, (x2 - 60, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def _draw_advanced_stats_panel(self, frame: np.ndarray, processed_data: Dict, w: int, h: int) -> np.ndarray:
        """Отрисовка расширенной панели статистики"""
        vehicles_data = processed_data['vehicles_data']
        processing_stats = processed_data.get('processing_stats', {})
        tracking_quality = processed_data.get('tracking_quality', 0)
        calibration_conf = processed_data.get('calibration_confidence', 0)

        # Создание фона для статистики
        stats_height = 180
        stats_bg = np.zeros((stats_height, 600, 3), dtype=np.uint8)
        frame[10:10 + stats_height, 10:610] = stats_bg

        active_vehicles = len(vehicles_data)
        total_tracked = len(self.tracker.get_all_analyzed_vehicles())
        aggressive_count = len(self.aggressive_drivers)

        # Расчет расширенной статистики скоростей
        speeds = []
        reliable_speeds = []
        for vehicle in self.tracker.get_active_vehicles():
            if vehicle.speeds:
                speeds.append(vehicle.last_reliable_speed)
                if vehicle.speed_confidence > 0.7:
                    reliable_speeds.append(vehicle.last_reliable_speed)

        avg_speed = np.mean(speeds) if speeds else 0
        avg_reliable_speed = np.mean(reliable_speeds) if reliable_speeds else 0

        # Отображение статистики
        y_offset = 35
        line_height = 25

        stats_lines = [
            f"Active Vehicles: {active_vehicles} | Total Tracked: {total_tracked}",
            f"Aggressive Drivers: {aggressive_count} | Speed: {avg_speed:.1f} km/h",
            f"Reliable Speed: {avg_reliable_speed:.1f} km/h | Tracking Quality: {tracking_quality:.2f}",
            f"Calibration: {calibration_conf:.2f} | Real FPS: {self.real_fps:.1f}",
            f"Processing: {processing_stats.get('avg_processing_time', 0) * 1000:.1f}ms | Frames: {self.frame_count}"
        ]

        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (20, y_offset + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Прогресс-бар качества трекинга
        bar_width = 200
        bar_height = 10
        bar_x, bar_y = 20, y_offset + len(stats_lines) * line_height + 10

        # Фон прогресс-бара
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Заполнение прогресс-бара
        fill_width = int(bar_width * tracking_quality)
        if tracking_quality > 0.7:
            bar_color = (0, 255, 0)
        elif tracking_quality > 0.4:
            bar_color = (0, 255, 255)
        else:
            bar_color = (0, 0, 255)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)

        cv2.putText(frame, f"Tracking Quality: {tracking_quality:.1%}",
                    (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def _update_extended_statistics(self, processed_data: Dict):
        """Обновление расширенной статистики"""
        current_hour = time.strftime("%H:00")
        self.extended_stats['time_distribution'][current_hour] += 1

        for vehicle_data in processed_data['vehicles_data']:
            vehicle_id = vehicle_data['id']
            speed = vehicle_data.get('speed', 0)

            if speed > 0:
                self.speed_stats.append(speed)

            if vehicle_data['is_aggressive']:
                self.aggressive_drivers.add(vehicle_id)

            for violation in vehicle_data['violations']:
                self.violation_stats[violation] += 1

        # Обновление пикового количества транспортных средств
        active_count = len(processed_data['vehicles_data'])
        if active_count > self.extended_stats['peak_vehicles']:
            self.extended_stats['peak_vehicles'] = active_count

    def generate_advanced_statistics_report(self, output_file: str = 'advanced_statistics.json'):
        """Генерация расширенного отчета статистики"""
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

        # Расширенная статистика транспортных средств
        vehicle_types = defaultdict(int)
        aggression_scores = []
        reliable_speeds = []

        for vehicle in all_vehicles:
            vehicle_types[vehicle.vehicle_class] += 1
            if hasattr(vehicle, 'last_reliable_speed') and vehicle.last_reliable_speed > 0:
                reliable_speeds.append(vehicle.last_reliable_speed)

        # Статистика нарушений
        violation_percentages = {}
        for viol_key, viol_name in self.VIOLATION_CATEGORIES.items():
            count = self.violation_stats.get(viol_key, 0)
            pct = (count / max(total_tracked, 1)) * 100
            violation_percentages[viol_name] = {
                'count': count,
                'percentage': round(pct, 2)
            }

        # Продвинутая статистика скоростей
        speed_stats = {}
        if self.speed_stats:
            speed_stats = {
                'average_speed': round(np.mean(self.speed_stats), 2),
                'median_speed': round(np.median(self.speed_stats), 2),
                'min_speed': round(np.min(self.speed_stats), 2),
                'max_speed': round(np.max(self.speed_stats), 2),
                'speed_std': round(np.std(self.speed_stats), 2),
                'reliable_speeds_count': len(reliable_speeds),
                'speed_distribution': {
                    '0-60 km/h': len([s for s in self.speed_stats if s < 60]),
                    '60-70 km/h': len([s for s in self.speed_stats if 60 <= s < 70]),
                    '70-80 km/h': len([s for s in self.speed_stats if 70 <= s < 80]),
                    '80-90 km/h': len([s for s in self.speed_stats if 80 <= s < 90]),
                    '90-100 km/h': len([s for s in self.speed_stats if 90 <= s < 100]),
                    '100-110 km/h': len([s for s in self.speed_stats if 100 <= s < 110]),
                    '110-120 km/h': len([s for s in self.speed_stats if 110 <= s < 120]),
                    '120+ km/h': len([s for s in self.speed_stats if s >= 120])
                }
            }

        # Качество системы
        system_quality = {
            'tracking_quality': round(self.tracker.get_tracking_quality(), 3),
            'calibration_confidence': round(self.tracker.calibration_data['calibration_confidence'], 3),
            'calibration_samples': self.tracker.calibration_data['calibration_samples'],
            'total_processing_time': round(time.time() - self.analysis_start_time, 2),
            'average_processing_fps': round(self.real_fps, 2)
        }

        report = {
            'analysis_summary': {
                'total_vehicles_tracked': total_tracked,
                'aggressive_drivers': aggressive_count,
                'normal_drivers': normal_count,
                'aggressive_percentage': round(aggressive_pct, 2),
                'normal_percentage': round(normal_pct, 2),
                'peak_vehicles_simultaneous': self.extended_stats['peak_vehicles'],
                'total_frames_processed': self.frame_count,
                'analysis_duration_seconds': round(self.frame_count / self.fps, 2),
                'video_fps': round(self.fps, 2),
                'real_processing_fps': round(self.real_fps, 2)
            },
            'speed_analysis': speed_stats,
            'vehicle_type_distribution': dict(vehicle_types),
            'violations_breakdown': violation_percentages,
            'system_quality': system_quality,
            'time_distribution': dict(self.extended_stats['time_distribution']),
            'analysis_parameters': {
                'detection_confidence': 0.55,
                'tracking_max_age_seconds': 5.0,
                'speed_calibration': 'adaptive',
                'calibration_reference_speed': '85 km/h',
                'aggression_threshold': '40%',
                'kalman_filter_enabled': True
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._print_advanced_report(report, total_tracked, aggressive_count, normal_count)

    def _print_advanced_report(self, report: Dict, total_tracked: int, aggressive_count: int, normal_count: int):
        """Вывод расширенного отчета в консоль"""
        print("\n" + "=" * 80)
        print(" " * 25 + "🚗 ADVANCED TRAFFIC ANALYSIS REPORT 🚗")
        print("=" * 80)

        summary = report['analysis_summary']
        speed_analysis = report['speed_analysis']
        system_quality = report['system_quality']

        print(f"\n📈 ANALYSIS SUMMARY:")
        print("-" * 50)
        print(f"   Total vehicles tracked: {total_tracked}")
        print(f"   Normal driving: {normal_count} ({summary['normal_percentage']:.1f}%)")
        print(f"   Aggressive driving: {aggressive_count} ({summary['aggressive_percentage']:.1f}%)")
        print(f"   Peak vehicles: {summary['peak_vehicles_simultaneous']}")
        print(f"   Frames processed: {summary['total_frames_processed']}")
        print(f"   Duration: {summary['analysis_duration_seconds']:.1f}s")

        print(f"\n⚡ SPEED ANALYSIS:")
        print("-" * 50)
        if speed_analysis:
            print(f"   Average speed: {speed_analysis['average_speed']:.1f} km/h")
            print(f"   Median speed: {speed_analysis['median_speed']:.1f} km/h")
            print(f"   Speed range: {speed_analysis['min_speed']:.1f} - {speed_analysis['max_speed']:.1f} km/h")
            print(f"   Speed deviation: ±{speed_analysis['speed_std']:.1f} km/h")
            print(f"   Reliable measurements: {speed_analysis['reliable_speeds_count']}")

            print(f"\n   📊 Speed Distribution:")
            for range_name, count in speed_analysis['speed_distribution'].items():
                if count > 0:
                    pct = (count / len(self.speed_stats)) * 100
                    bar = "█" * int(pct / 5)
                    print(f"     {range_name:<12} {bar} {count:3d} vehicles ({pct:5.1f}%)")

        print(f"\n🎯 SYSTEM QUALITY:")
        print("-" * 50)
        print(f"   Tracking quality: {system_quality['tracking_quality']:.1%}")
        print(f"   Calibration confidence: {system_quality['calibration_confidence']:.1%}")
        print(f"   Calibration samples: {system_quality['calibration_samples']}")
        print(f"   Processing FPS: {system_quality['average_processing_fps']:.1f}")
        print(f"   Total time: {system_quality['total_processing_time']:.1f}s")

        print(f"\n🚨 VIOLATION TYPES:")
        print("-" * 50)
        violations = report['violations_breakdown']
        sorted_violations = sorted(violations.items(), key=lambda x: x[1]['count'], reverse=True)

        for viol_name, data in sorted_violations:
            if data['count'] > 0:
                bar = "█" * int(data['percentage'] / 3)
                print(f"   {viol_name:<25} {bar} {data['count']:>4} cases ({data['percentage']:>5.1f}%)")

        print("\n" + "=" * 80)
        print("   🎉 Analysis completed successfully!")
        print("=" * 80)


def main():
    VIDEO_PATH = "input.mp4"
    OUTPUT_VIDEO = "advanced_output.mp4"
    STATS_JSON = "advanced_traffic_statistics.json"

    try:
        print("🚀 Starting Advanced Traffic Analyzer with Enhanced Speed Detection...")
        print("🔧 Initializing advanced systems...")

        # Проверка доступности модели
        if not os.path.exists('yolo11x.pt'):
            print("❌ YOLO model not found. Please download yolo11x.pt")
            return

        analyzer = AdvancedTrafficAnalyzer(VIDEO_PATH)
        analyzer.process_video(OUTPUT_VIDEO, STATS_JSON)

    except FileNotFoundError:
        print("❌ Error: Video file not found! Please check the path.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

