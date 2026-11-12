import os
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

NORM_SIZE = 256

YOLO_WEIGHTS_PATH = "../runs/detect/train/weights/best.pt"
FRAME_SAMPLE_PERIOD_SEC = 2.0  


def to_angle_360(vx, vy):
    angle_deg = math.degrees(math.atan2(vy, vx))
    return (angle_deg + 360.0) % 360.0   


def preprocess_tach_image(img, norm_size=NORM_SIZE):
    img_resized = cv2.resize(img, (norm_size, norm_size), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Выравнивание освещённости
    gamma = 0.9
    gray_norm = gray.astype(np.float32) / 255.0
    gray_gamma = np.power(gray_norm, gamma)
    gray_gamma = np.uint8(np.clip(gray_gamma * 255.0, 0, 255))

    # Локальный контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_gamma)

    # Шум
    gray_med = cv2.medianBlur(gray_clahe, 3)

    # Повышение резкости
    kernel_sharp = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float32
    )
    gray_sharp = cv2.filter2D(gray_med, -1, kernel_sharp)

    edges = cv2.Canny(
        gray_sharp,
        threshold1=50,
        threshold2=150,
        apertureSize=3,
        L2gradient=True
    )

    return img_resized, edges


def calibrate_min_angle(img):
    """
    Калибровка: клик по минимальной риске (0) на КРОПЕ тахометра.
    """
    img_norm, _ = preprocess_tach_image(img, norm_size=NORM_SIZE)

    h, w = img_norm.shape[:2]
    cx, cy = w // 2, h // 2

    # Рисуем начальное изображение с центральной точкой
    display = img_norm.copy()
    cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)

    click_pt = [None]

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt[0] = (x, y)

    win_name = "Calibration"

    print("КАЛИБРОВКА MIN:")
    print("  Кликни по риске МИНИМАЛЬНОГО значения (обычно 0) на КРОПЕ тахометра.")
    print("  ESC или 'q' — отмена.\n")

    cv2.imshow(win_name, display)
    cv2.setMouseCallback(win_name, on_mouse)

    while click_pt[0] is None:
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            cv2.destroyWindow(win_name)
            print("Калибровка отменена пользователем.")
            return None

    # Показываем финальное изображение с отметкой клика
    x, y = click_pt[0]
    final_display = img_norm.copy()
    cv2.circle(final_display, (cx, cy), 4, (0, 255, 0), -1)
    cv2.circle(final_display, (x, y), 4, (0, 0, 255), -1)
    cv2.line(final_display, (cx, cy), (x, y), (0, 255, 255), 2)
    cv2.imshow(win_name, final_display)
    cv2.waitKey(500)
    cv2.destroyWindow(win_name)

    vx = x - cx
    vy = y - cy
    theta_min_360 = to_angle_360(vx, vy)

    print(f"Угол минимальной риски (0) = {theta_min_360:.2f}°\n")
    return theta_min_360


def detect_needle_angle_360(img, debug_draw=False):
    """
    Находит угол стрелки тахометра в [0, 360).

    Возвращает:
      theta_needle_360 – угол стрелки в [0, 360)
      img_visual       – картинка с нарисованной найденной стрелкой (если debug_draw=True)
    """
    img_norm, edges = preprocess_tach_image(img, norm_size=NORM_SIZE)
    h, w = img_norm.shape[:2]
    cx, cy = w // 2, h // 2

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=int(min(h, w) * 0.25),
        maxLineGap=int(min(h, w) * 0.05)
    )

    if lines is None:
        print("HoughLinesP не нашёл ни одной линии.")
        return None, img_norm

    best_score = -1.0
    best_line = None

    max_dist_center = min(h, w) * 0.2   # линия должна проходить относительно близко к центру
    min_radial_cos = 0.9                # степень "радиальности"

    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < max_dist_center:
            continue

        # расстояние от центра до прямой
        dist_line = abs(dx * (cy - y1) - (cx - x1) * dy) / (length + 1e-6)
        if dist_line > max_dist_center:
            continue

        # середина линии
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        rvx = mx - cx
        rvy = my - cy
        r_len = math.hypot(rvx, rvy)
        if r_len < 1e-6:
            continue

        # косинус угла между направлением линии и радиальным направлением
        d_len = length
        cos_phi = abs((dx * rvx + dy * rvy) / (d_len * r_len))
        if cos_phi < min_radial_cos:
            continue

        score = length * cos_phi / (1.0 + dist_line)
        if score > best_score:
            best_score = score
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        print("Не удалось найти подходящую радиальную линию для стрелки.")
        return None, img_norm

    x1, y1, x2, y2 = best_line

    # выбираем кончик — конец дальше от центра
    d1 = math.hypot(x1 - cx, y1 - cy)
    d2 = math.hypot(x2 - cx, y2 - cy)
    if d1 > d2:
        tip_x, tip_y = x1, y1
    else:
        tip_x, tip_y = x2, y2

    vx = tip_x - cx
    vy = tip_y - cy

    theta_needle_360 = to_angle_360(vx, vy)

    img_visual = img_norm.copy()
    if debug_draw:
        cv2.circle(img_visual, (cx, cy), 4, (0, 255, 0), -1)
        cv2.circle(img_visual, (tip_x, tip_y), 4, (0, 0, 255), -1)
        cv2.line(img_visual, (cx, cy), (tip_x, tip_y), (0, 255, 255), 2)
        text = f"{theta_needle_360:.1f} deg"
        cv2.putText(img_visual, text, (tip_x + 5, tip_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    print(f"Угол стрелки (0–360): {theta_needle_360:.2f}°")
    return theta_needle_360, img_visual


def compute_tach_value_from_angles(theta_min_360, theta_needle_360, max_value):
    """
    Симметрия шкалы относительно OY и рост значений по часовой.
    """
    # симметричный угол максимальной риски
    theta_max_360 = (180.0 - theta_min_360) % 360.0

    # полная дуга шкалы по часовой
    arc_total = 360 - ((theta_min_360 - theta_max_360 + 360.0) % 360.0)

    # дуга от min до стрелки по часовой
    arc_needle = theta_needle_360 - theta_min_360

    print(f"theta_min = {theta_min_360:.2f}°, theta_max = {theta_max_360:.2f}°")
    print(f"arc_total  (min->max по часовой)    = {arc_total:.2f}°")
    print(f"arc_needle (min->needle по часовой) = {arc_needle:.2f}°")

    if arc_total < 1e-3:
        raise ValueError("Слишком малая дуга шкалы – проверь калибровку.")

    # если стрелка явно вне дуги шкалы – считаем, что она "ниже нуля"
    if arc_needle > arc_total:
        print("Стрелка вне рабочей дуги (по часовой от min до max)")
        return 0.0

    ratio = arc_needle / arc_total  # [0..1]
    value = max_value * ratio
    if value > 0:
        return value
    else:
        return 0


def collect_detections_in_region(r, class_names, region=None, max_items=8):
    """
    Возвращает список строк с детекциями:
      - если region=None — по всему кадру,
      - если region=(x1,y1,x2,y2) — только боксы, центр которых попадает внутрь.
    """
    lines = []

    if r.boxes is None or len(r.boxes) == 0:
        return lines

    rx1 = ry1 = rx2 = ry2 = None
    if region is not None:
        rx1, ry1, rx2, ry2 = region

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        if region is not None:
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue

        name = class_names.get(cls_id, str(cls_id))
        lines.append(f"{name}: {conf:.2f}")
        if len(lines) >= max_items:
            break

    return lines


def make_info_panel(value, det_full, det_tach, det_speed, height, infer_time=None):
    """
    Рисует чёрную панель справа от кропа и выводит туда текст:
      - значение тахометра
      - время инференса
      - список детекций на кадре.
    (det_tach и det_speed сейчас не выводим, но оставляем в сигнатуре.)
    """
    panel_width = 350
    panel = np.zeros((height, panel_width, 3), dtype=np.uint8)

    y = 25
    line_h = 20

    def put_line(text, color=(0, 255, 255), bold=False):
        nonlocal y
        if y > height - 10:
            return
        thickness = 2 if bold else 1
        cv2.putText(panel, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, thickness, cv2.LINE_AA)
        y += line_h

    put_line("Tachometer value:", bold=True)
    put_line(f"{value:.2f}", color=(0, 255, 0), bold=True)
    y += line_h // 2

    if infer_time is not None:
        put_line("Inference time:", bold=True)
        put_line(f"{infer_time*1000:.1f} ms", color=(255, 255, 255))
        y += line_h // 2

    put_line("Detections (frame):", bold=True)
    if det_full:
        for t in det_full:
            put_line(f"- {t}")
    else:
        put_line("- none")
    y += line_h // 2

    return panel


def sample_frames_from_video(path, period_sec):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("[ERROR] Не удалось открыть видео.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_interval = max(int(round(fps * period_sec)), 1)

    frames = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            t_sec = frame_idx / fps
            frames.append((frame.copy(), t_sec, frame_idx))
        frame_idx += 1

    cap.release()
    print(f"Извлечено {len(frames)} кадров (каждые ~{period_sec} сек)\n")
    return frames


# -------------------- РЕЖИМ ВИДЕО -------------------- #

def run_video_mode(video_path):
    # фреймы
    frames = sample_frames_from_video(video_path, FRAME_SAMPLE_PERIOD_SEC)
    if not frames:
        return

    # YOLO
    model = YOLO(YOLO_WEIGHTS_PATH)

    # имена классов
    if isinstance(model.names, dict):
        class_names = model.names
    else:
        class_names = {i: n for i, n in enumerate(model.names)}

    # находим id тахометра и спидометра
    tach_id = None
    speed_id = None
    for cid, name in class_names.items():
        if name == "tachometer":
            tach_id = cid
        if name == "speedometer":
            speed_id = cid

    if tach_id is None:
        print("[ERROR] Класс 'tachometer' не найден в model.names")
        return

    # макс значение тахометра
    try:
        max_value = float(input("Введите максимальное значение тахометра (например, 60 или 8000): "))
    except Exception as e:
        print("Ошибка ввода max:", e)
        return

    theta_min_360 = None  # калибровка min — один раз

    for frame_bgr, t_sec, frame_idx in frames:
        print(f"\n=== КАДР #{frame_idx} (t = {t_sec:.2f} с) ===")

        # YOLO-инференс + замер времени
        t0 = time.time()
        results = model(frame_bgr, conf=0.3)
        infer_time = time.time() - t0  # секунды
        r = results[0]

        # ищем bbox тахометра и спидометра
        tach_boxes = []
        speed_boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if cls_id == tach_id:
                    tach_boxes.append((conf, (int(x1), int(y1), int(x2), int(y2))))
                if speed_id is not None and cls_id == speed_id:
                    speed_boxes.append((conf, (int(x1), int(y1), int(x2), int(y2))))

        if not tach_boxes:
            print("Тахометр на этом кадре не найден.")
            continue

        # самый уверенный тахометр
        tach_boxes.sort(key=lambda x: x[0], reverse=True)
        best_conf, (x1, y1, x2, y2) = tach_boxes[0]

        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            print("Кривые координаты bbox, пропускаем кадр.")
            continue

        tach_crop = frame_bgr[y1:y2, x1:x2].copy()
        print(f"Нашли тахометр (conf={best_conf:.2f}), bbox=({x1},{y1})-({x2},{y2})")

        # калибровка по ПЕРВОМУ тахометру
        if theta_min_360 is None:
            theta_min_360 = calibrate_min_angle(tach_crop)
            if theta_min_360 is None:
                print("Калибровка не выполнена, прекращаем.")
                return

        # угол стрелки
        theta_needle_360, vis = detect_needle_angle_360(tach_crop, debug_draw=True)
        if theta_needle_360 is None:
            print("Не удалось определить угол стрелки на этом кадре.")
            continue

        # значение тахометра
        value = compute_tach_value_from_angles(theta_min_360, theta_needle_360, max_value)
        print(f"Оценка значения тахометра: {value:.2f}")
        print(f"[DEBUG] Inference time: {infer_time*1000:.1f} ms")

        # ----------- собираем детекции для показа в панели ----------- #
        det_full = collect_detections_in_region(r, class_names, region=None)
        det_tach = collect_detections_in_region(r, class_names, region=(x1, y1, x2, y2))
        det_speed = None
        if speed_boxes:
            speed_boxes.sort(key=lambda x: x[0], reverse=True)
            sp_conf, (sx1, sy1, sx2, sy2) = speed_boxes[0]
            sx1 = max(0, min(sx1, w - 1))
            sx2 = max(0, min(sx2, w - 1))
            sy1 = max(0, min(sy1, h - 1))
            sy2 = max(0, min(sy2, h - 1))
            det_speed = collect_detections_in_region(r, class_names, region=(sx1, sy1, sx2, sy2))

        # делаем панель справа и объединяем
        vis_bgr = vis
        if len(vis_bgr.shape) == 2:
            vis_bgr = cv2.cvtColor(vis_bgr, cv2.COLOR_GRAY2BGR)

        info_panel = make_info_panel(value, det_full, det_tach, det_speed,
                                     vis_bgr.shape[0], infer_time=infer_time)
        ui_image = np.hstack([vis_bgr, info_panel])

        cv2.imshow("Tachometer UI (video)", ui_image)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()


# -------------------- РЕЖИМ ИЗОБРАЖЕНИЯ -------------------- #

def run_image_mode(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Не удалось загрузить изображение: {image_path}")
        return

    # YOLO
    model = YOLO(YOLO_WEIGHTS_PATH)

    # имена классов
    if isinstance(model.names, dict):
        class_names = model.names
    else:
        class_names = {i: n for i, n in enumerate(model.names)}

    # находим id тахометра и спидометра
    tach_id = None
    speed_id = None
    for cid, name in class_names.items():
        if name == "tachometer":
            tach_id = cid
        if name == "speedometer":
            speed_id = cid

    if tach_id is None:
        print("[ERROR] Класс 'tachometer' не найден в model.names")
        return

    # макс значение тахометра
    try:
        max_value = float(input("Введите максимальное значение тахометра (например, 60 или 8000): "))
    except Exception as e:
        print("Ошибка ввода max:", e)
        return

    # YOLO-инференс + замер времени
    t0 = time.time()
    results = model(img, conf=0.3)
    infer_time = time.time() - t0
    r = results[0]

    # ищем bbox тахометра и спидометра
    tach_boxes = []
    speed_boxes = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if cls_id == tach_id:
                tach_boxes.append((conf, (int(x1), int(y1), int(x2), int(y2))))
            if speed_id is not None and cls_id == speed_id:
                speed_boxes.append((conf, (int(x1), int(y1), int(x2), int(y2))))

    if not tach_boxes:
        print("Тахометр на изображении не найден.")
        return

    # самый уверенный тахометр
    tach_boxes.sort(key=lambda x: x[0], reverse=True)
    best_conf, (x1, y1, x2, y2) = tach_boxes[0]

    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        print("Кривые координаты bbox, прекращаем.")
        return

    tach_crop = img[y1:y2, x1:x2].copy()
    print(f"Нашли тахометр (conf={best_conf:.2f}), bbox=({x1},{y1})-({x2},{y2})")

    # калибровка по этому тахометру
    theta_min_360 = calibrate_min_angle(tach_crop)
    if theta_min_360 is None:
        print("Калибровка не выполнена.")
        return

    # угол стрелки
    theta_needle_360, vis = detect_needle_angle_360(tach_crop, debug_draw=True)
    if theta_needle_360 is None:
        print("Не удалось определить угол стрелки.")
        return

    # значение тахометра
    value = compute_tach_value_from_angles(theta_min_360, theta_needle_360, max_value)
    print(f"Оценка значения тахометра: {value:.2f}")
    print(f"[DEBUG] Inference time: {infer_time*1000:.1f} ms")

    # ----------- собираем детекции для панели ----------- #
    det_full = collect_detections_in_region(r, class_names, region=None)
    det_tach = collect_detections_in_region(r, class_names, region=(x1, y1, x2, y2))
    det_speed = None
    if speed_boxes:
        speed_boxes.sort(key=lambda x: x[0], reverse=True)
        sp_conf, (sx1, sy1, sx2, sy2) = speed_boxes[0]
        sx1 = max(0, min(sx1, w - 1))
        sx2 = max(0, min(sx2, w - 1))
        sy1 = max(0, min(sy1, h - 1))
        sy2 = max(0, min(sy2, h - 1))
        det_speed = collect_detections_in_region(r, class_names, region=(sx1, sy1, sx2, sy2))

    vis_bgr = vis
    if len(vis_bgr.shape) == 2:
        vis_bgr = cv2.cvtColor(vis_bgr, cv2.COLOR_GRAY2BGR)

    info_panel = make_info_panel(value, det_full, det_tach, det_speed,
                                 vis_bgr.shape[0], infer_time=infer_time)
    ui_image = np.hstack([vis_bgr, info_panel])

    cv2.imshow("Tachometer UI (image)", ui_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------- MAIN -------------------- #

def main():
    mode = input("Режим: 'video' или 'img' (ENTER = video): ").strip().lower()

    if mode == "img":
        image_path = input("Введите путь к изображению (ENTER для 'input.png'): ").strip()
        if not image_path:
            image_path = "input.png"
        run_image_mode(image_path)
    else:
        video_path = input("Введите путь к видео (ENTER для 'input.mp4'): ").strip()
        if not video_path:
            video_path = "input.mp4"
        run_video_mode(video_path)


if __name__ == "__main__":
    main()
