import cv2
import numpy as np

def principal_axes_from_contour(cnt):
    """
    Возвращает:
      mean (2,), eigenvalues (2,), eigenvectors (2,2)
    где eigenvectors[:,0] — главная ось (для max eigenvalue).
    """
    pts = cnt.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)

    # ковариация (PCA). Можно через моменты, но так устойчивее и проще.
    centered = pts - mean
    cov = (centered.T @ centered) / max(len(pts), 1)

    # eigen decomposition (2x2)
    vals, vecs = np.linalg.eigh(cov)  # vals по возрастанию
    order = np.argsort(vals)[::-1]    # по убыванию
    vals = vals[order]
    vecs = vecs[:, order]
    return mean, vals, vecs

def contour_endpoints_along_axis(cnt, mean, axis_vec):
    """
    Находит "концы" контура как min/max проекцию на главную ось.
    """
    pts = cnt.reshape(-1, 2).astype(np.float64)
    proj = (pts - mean) @ axis_vec  # скалярная проекция
    p_min = pts[np.argmin(proj)]
    p_max = pts[np.argmax(proj)]
    return p_min, p_max, proj.max() - proj.min()

def point_line_distance(pt, line_point, line_dir_unit):
    """
    Расстояние от точки pt до бесконечной прямой (line_point + t*line_dir_unit)
    """
    v = pt - line_point
    # в 2D: |v - (v·d)d|
    proj = np.dot(v, line_dir_unit) * line_dir_unit
    perp = v - proj
    return np.linalg.norm(perp)

def find_needle_contour(bgr, debug=False):
    h, w = bgr.shape[:2]
    center = np.array([w/2.0, h/2.0], dtype=np.float64)
    diag = np.hypot(w, h)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 250, 400)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # --- 2) Контуры ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best_cnt = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # отсекаем мелочь (подкрутить под ваши изображения)
            continue

        mean, vals, vecs = principal_axes_from_contour(cnt)
        lam1, lam2 = vals[0], vals[1]

        # Вытянутость: чем больше, тем "линейнее"
        elong = (lam1 + 1e-9) / (lam2 + 1e-9)

        # Главная ось
        axis = vecs[:, 0]
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        # "Длина" по главной оси + концы
        p1, p2, length = contour_endpoints_along_axis(cnt, mean, axis)

        # Условие "стрелка выходит из центра": один конец близко к центру
        d1 = np.linalg.norm(p1 - center)
        d2 = np.linalg.norm(p2 - center)
        d_end = min(d1, d2)

        # Доп. проверка: центр рядом с главной осью (стрелка должна проходить рядом)
        d_line = point_line_distance(center, mean, axis)

        # Нормируем метрики
        length_n = length / (diag + 1e-9)
        d_end_n = d_end / (diag + 1e-9)
        d_line_n = d_line / (diag + 1e-9)

        # Гейты (жесткие отсекающие условия) — можно ослабить/усилить
        if elong < 10:            # недостаточно вытянут
            continue
        if length_n < 0.15:       # слишком короткий объект
            continue
        if d_end_n > 0.20:        # ни один конец не близко к центру
            continue
        if d_line_n > 0.10:       # главная ось далеко от центра
            continue

        # Score: хотим длинную, очень вытянутую, с концом у центра
        score = (np.log(elong) * 2.0) + (length_n * 3.0) + ((0.20 - d_end_n) * 4.0) + ((0.10 - d_line_n) * 2.0)

        if score > best_score:
            best_score = score
            best_cnt = cnt

    if debug:
        dbg = bgr.copy()
        if best_cnt is not None:
            cv2.drawContours(dbg, [best_cnt], -1, (0, 255, 0), 2)
        return best_cnt, edges, dbg

    return best_cnt


img = cv2.imread("test4.png")
if img is None:
    raise FileNotFoundError("Не удалось прочитать test.png (проверь путь/имя файла)")

cnt = find_needle_contour(img, debug=False)

# Маска и визуализация результата
mask = np.zeros(img.shape[:2], np.uint8)

if cnt is not None:
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # рисуем контур на копии исходного изображения
    overlay = img.copy()
    cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), thickness=2)

    # Показать окна (работает в обычном Python/IDE)
    cv2.imshow("needle mask", mask)
    cv2.imshow("needle contour", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # На всякий случай сохранить
    cv2.imwrite("needle_mask.png", mask)
    cv2.imwrite("needle_overlay.png", overlay)
else:
    print("Контур стрелки не найден (cnt is None).")

    # всё равно покажем пустую маску и исходник
    cv2.imshow("needle mask", mask)
    cv2.imshow("input", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("needle_mask.png", mask)
    cv2.imwrite("input.png", img)

