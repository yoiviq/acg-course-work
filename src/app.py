import os
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

NORM_SIZE = 256

YOLO_WEIGHTS_PATH = "../runs/detect/train/weights/best.pt"
FRAME_SAMPLE_PERIOD_SEC = 2.0
STANDARD_VIDEO_PATH = "../videos/input.mp4"


class App:
    def __init__(self, model, mode, path):
        self.modelPath = model
        self.mode = (mode or "").strip().lower()
        self.path = (path or "").strip()
        self.needleMethod = "HL"
        self.frameExtractTime = None

    def start(self):
        self.initModelAndClasses()

        self.needleMethod = self.askNeedleMethod()
        maxValue = self.askMaxValue()
        if maxValue is None:
            return

        if self.mode == "video":
            frames = self.getFrames()
            if not frames:
                return
            self.runVideoFrames(frames, maxValue)
        else:
            self.processImage(maxValue)

    def getFrames(self):
        t0 = time.time()
        frames = self.sampleFramesFromVideo(self.path, FRAME_SAMPLE_PERIOD_SEC)
        self.frameExtractTime = time.time() - t0
        return frames

    def processImage(self, maxValue):
        img = cv2.imread(self.path)
        if img is None:
            print(f"[ERROR] Не удалось загрузить изображение: {self.path}")
            return

        t0 = time.time()
        results = self.model(img, conf=0.3)
        inferTime = time.time() - t0
        r = results[0]

        tachBoxes = self.extractTachBoxes(r)

        if not tachBoxes:
            print("Тахометр на изображении не найден.")
            return

        bestConf, box = self.pickBestBox(tachBoxes, img.shape[:2])
        if box is None:
            print("Кривые координаты bbox, прекращаем.")
            return

        x1, y1, x2, y2 = box
        tachCrop = img[y1:y2, x1:x2].copy()
        print(f"Нашли тахометр (conf={bestConf:.2f}), bbox=({x1},{y1})-({x2},{y2})")

        thetaMin360 = self.calibrateMinAngle(tachCrop)
        if thetaMin360 is None:
            print("Калибровка не выполнена.")
            return

        tN = time.time()
        thetaNeedle360, vis = self.detectNeedleAngle360(tachCrop, debugDraw=True)
        needleTime = time.time() - tN
        if thetaNeedle360 is None:
            print("Не удалось определить угол стрелки.")
            return

        value = self.computeTachValueFromAngles(thetaMin360, thetaNeedle360, maxValue)
        print(f"Оценка значения тахометра: {value:.2f}")
        print(f"[DEBUG] Inference time: {inferTime * 1000:.1f} ms")

        detFull = self.collectDetectionsInRegion(r, region=None)
        detTach = self.collectDetectionsInRegion(r, region=(x1, y1, x2, y2))

        infoPanel = self.makeInfoPanel(
            value, detFull, detTach, vis.shape[0],
            inferTime=inferTime,
            needleTime=needleTime,
            frameExtractTime=None
        )
        uiImage = np.hstack([vis, infoPanel])

        cv2.imshow("Tachometer UI (image)", uiImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def initModelAndClasses(self):
        self.model = YOLO(self.modelPath)

        if isinstance(self.model.names, dict):
            self.classNames = self.model.names
        else:
            self.classNames = {i: n for i, n in enumerate(self.model.names)}

        self.tachId = None
        for cid, name in self.classNames.items():
            if name == "tachometer":
                self.tachId = cid

        if self.tachId is None:
            raise RuntimeError("[ERROR] Класс 'tachometer' не найден в model.names")

    def askNeedleMethod(self):
        s = input("Метод детекции стрелки (HL или PCA) [HL]: ").strip().upper()
        if not s:
            return "HL"
        if s in ("HL", "H", "HOUGH"):
            return "HL"
        if s in ("PCA", "P"):
            return "PCA"
        print("Не понял ввод, использую HL.")
        return "HL"

    def askMaxValue(self):
        try:
            return float(input("Введите максимальное значение тахометра (например, 60 или 8000): "))
        except Exception as e:
            print("Ошибка ввода max:", e)
            return None

    def runVideoFrames(self, frames, maxValue):
        thetaMin360 = None

        for frameBgr, tSec, frameIdx in frames:
            print(f"\n=== КАДР #{frameIdx} (t = {tSec:.2f} с) ===")

            t0 = time.time()
            results = self.model(frameBgr, conf=0.3)
            inferTime = time.time() - t0
            r = results[0]

            tachBoxes = self.extractTachBoxes(r)

            if not tachBoxes:
                print("Тахометр на этом кадре не найден.")
                continue

            bestConf, box = self.pickBestBox(tachBoxes, frameBgr.shape[:2])
            if box is None:
                print("Кривые координаты bbox, пропускаем кадр.")
                continue

            x1, y1, x2, y2 = box
            tachCrop = frameBgr[y1:y2, x1:x2].copy()
            print(f"Нашли тахометр (conf={bestConf:.2f})")

            if thetaMin360 is None:
                thetaMin360 = self.calibrateMinAngle(tachCrop)
                if thetaMin360 is None:
                    print("Калибровка не выполнена, прекращаем.")
                    return

            tN = time.time()
            thetaNeedle360, vis = self.detectNeedleAngle360(tachCrop, debugDraw=True)
            needleTime = time.time() - tN
            if thetaNeedle360 is None:
                print("Не удалось определить угол стрелки на этом кадре.")
                continue

            value = self.computeTachValueFromAngles(thetaMin360, thetaNeedle360, maxValue)
            print(f"Оценка значения тахометра: {value:.2f}")
            print(f"[DEBUG] Inference time: {inferTime * 1000:.1f} ms")

            detFull = self.collectDetectionsInRegion(r, region=None)
            detTach = self.collectDetectionsInRegion(r, region=(x1, y1, x2, y2))

            infoPanel = self.makeInfoPanel(
                value, detFull, detTach, vis.shape[0],
                inferTime=inferTime,
                needleTime=needleTime,
                frameExtractTime=self.frameExtractTime
            )
            uiImage = np.hstack([vis, infoPanel])

            title = f"Tachometer UI (video) | needle={self.needleMethod}"
            cv2.imshow(title, uiImage)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(title)
            if key in (27, ord("q")):
                break

        cv2.destroyAllWindows()

    def toAngle360(self, vx, vy):
        angleDeg = math.degrees(math.atan2(vy, vx))
        return (angleDeg + 360.0) % 360.0

    def preprocessTachImage(self, img, normSize=NORM_SIZE):
        imgResized = cv2.resize(img, (normSize, normSize), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(imgResized, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grayClahe = clahe.apply(gray)

        grayMed = cv2.medianBlur(grayClahe, 3)

        kernelSharp = np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]],
            dtype=np.float32
        )
        graySharp = cv2.filter2D(grayMed, -1, kernelSharp)

        edges = cv2.Canny(
            graySharp,
            threshold1=250,
            threshold2=400,
            apertureSize=3,
        )

        return imgResized, edges

    def preprocessCropBgr(self, bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, bb = cv2.split(lab)

        gamma = 0.9
        lNorm = l.astype(np.float32) / 255.0
        lGamma = np.power(lNorm, gamma)
        lGamma = np.uint8(np.clip(lGamma * 255.0, 0, 255))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lClahe = clahe.apply(lGamma)

        lab2 = cv2.merge([lClahe, a, bb])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        out = cv2.medianBlur(out, 3)

        kernelSharp = np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]],
            dtype=np.float32
        )
        out = cv2.filter2D(out, -1, kernelSharp)
        return out

    def calibrateMinAngle(self, img):
        imgNorm, _ = self.preprocessTachImage(img, normSize=NORM_SIZE)

        h, w = imgNorm.shape[:2]
        cx, cy = w // 2, h // 2

        display = imgNorm.copy()
        cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)

        clickPt = [None]

        def onMouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                clickPt[0] = (x, y)

        winName = "Calibration"

        print("КАЛИБРОВКА MIN:")
        print("Кликните по риске минимального значения на тахометре.")
        print("ESC или 'q' — отмена.\n")

        cv2.imshow(winName, display)
        cv2.setMouseCallback(winName, onMouse)

        while clickPt[0] is None:
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord("q"):
                cv2.destroyWindow(winName)
                print("Калибровка отменена пользователем.")
                return None

        x, y = clickPt[0]
        finalDisplay = imgNorm.copy()
        cv2.circle(finalDisplay, (cx, cy), 4, (0, 255, 0), -1)
        cv2.circle(finalDisplay, (x, y), 4, (0, 0, 255), -1)
        cv2.line(finalDisplay, (cx, cy), (x, y), (0, 255, 255), 2)
        cv2.imshow(winName, finalDisplay)
        cv2.waitKey(500)
        cv2.destroyWindow(winName)

        vx = x - cx
        vy = y - cy
        thetaMin360 = self.toAngle360(vx, vy)

        print(f"Угол минимальной риски (0) = {thetaMin360:.2f}°\n")
        return thetaMin360

    def detectNeedleAngle360(self, img, debugDraw=False):
        if self.needleMethod == "PCA":
            return self.detectNeedleAngle360Pca(img, debugDraw=debugDraw)
        return self.detectNeedleAngle360Hl(img, debugDraw=debugDraw)

    def detectNeedleAngle360Hl(self, img, debugDraw=False):
        imgNorm, edges = self.preprocessTachImage(img, normSize=NORM_SIZE)

        h, w = imgNorm.shape[:2]
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
            return None, imgNorm

        bestScore = -1.0
        bestLine = None

        maxDistCenter = min(h, w) * 0.2
        minRadialCos = 0.9

        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < maxDistCenter:
                continue

            distLine = abs(dx * (cy - y1) - (cx - x1) * dy) / (length + 1e-6)
            if distLine > maxDistCenter:
                continue

            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)

            rvx = mx - cx
            rvy = my - cy
            rLen = math.hypot(rvx, rvy)
            if rLen < 1e-6:
                continue

            cosPhi = abs((dx * rvx + dy * rvy) / ((length + 1e-9) * (rLen + 1e-9)))
            if cosPhi < minRadialCos:
                continue

            score = length * cosPhi / (1.0 + distLine)
            if score > bestScore:
                bestScore = score
                bestLine = (x1, y1, x2, y2)

        if bestLine is None:
            print("Не удалось найти подходящую радиальную линию для стрелки.")
            return None, imgNorm

        x1, y1, x2, y2 = bestLine

        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        if d1 > d2:
            tipX, tipY = x1, y1
        else:
            tipX, tipY = x2, y2

        vx = tipX - cx
        vy = tipY - cy
        thetaNeedle360 = self.toAngle360(vx, vy)

        imgVisual = imgNorm.copy()
        if debugDraw:
            cv2.circle(imgVisual, (cx, cy), 4, (0, 255, 0), -1)
            cv2.circle(imgVisual, (tipX, tipY), 4, (0, 0, 255), -1)
            cv2.line(imgVisual, (cx, cy), (tipX, tipY), (0, 255, 255), 2)
            text = f"{thetaNeedle360:.1f} deg"
            cv2.putText(imgVisual, text, (tipX + 5, tipY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        print(f"Угол стрелки (0–360): {thetaNeedle360:.2f}°")
        return thetaNeedle360, imgVisual

    def principalAxesFromContour(self, cnt):
        pts = cnt.reshape(-1, 2).astype(np.float64)
        mean = pts.mean(axis=0)

        centered = pts - mean
        cov = (centered.T @ centered) / max(len(pts), 1)

        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        return mean, vals, vecs

    def contourEndpointsAlongAxis(self, cnt, mean, axisVec):
        pts = cnt.reshape(-1, 2).astype(np.float64)
        proj = (pts - mean) @ axisVec
        pMin = pts[np.argmin(proj)]
        pMax = pts[np.argmax(proj)]
        return pMin, pMax, proj.max() - proj.min()

    def pointLineDistance(self, pt, linePoint, lineDirUnit):
        v = pt - linePoint
        proj = np.dot(v, lineDirUnit) * lineDirUnit
        perp = v - proj
        return np.linalg.norm(perp)

    def findNeedleContour(self, bgr, debug=False):
        h, w = bgr.shape[:2]
        center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
        diag = np.hypot(w, h)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 250, 400)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        bestCnt = None
        bestScore = -1.0

        for cnt in contours:

            ptsTmp = cnt.reshape(-1, 2).astype(np.float64)
            dMin = np.min(np.hypot(ptsTmp[:, 0] - center[0], ptsTmp[:, 1] - center[1]))
            if dMin > (min(h, w) * 0.25):
                continue

            mean, vals, vecs = self.principalAxesFromContour(cnt)
            lam1, lam2 = float(vals[0]), float(vals[1])

            elong = (lam1 + 1e-9) / (lam2 + 1e-9)

            axis = vecs[:, 0]
            axis = axis / (np.linalg.norm(axis) + 1e-12)

            p1, p2, length = self.contourEndpointsAlongAxis(cnt, mean, axis)

            d1 = np.linalg.norm(p1 - center)
            d2 = np.linalg.norm(p2 - center)
            dEnd = min(d1, d2)

            dLine = self.pointLineDistance(center, mean, axis)

            lengthN = length / (diag + 1e-9)
            dEndN = dEnd / (diag + 1e-9)
            dLineN = dLine / (diag + 1e-9)

            if elong < 10:
                continue
            if lengthN < 0.15:
                continue

            score = (
                (np.log(elong) * 2.0)
                + (lengthN * 3.0)
                + ((0.20 - dEndN) * 4.0)
                + ((0.10 - dLineN) * 2.0)
            )

            if score > bestScore:
                bestScore = score
                bestCnt = cnt

        if debug:
            dbg = bgr.copy()
            if bestCnt is not None:
                cv2.drawContours(dbg, [bestCnt], -1, (0, 255, 0), 2)
            return bestCnt, edges, dbg

        return bestCnt, edges, None

    def detectNeedleAngle360Pca(self, img, debugDraw=False):
        imgNorm, _ = self.preprocessTachImage(img, normSize=NORM_SIZE)

        h, w = imgNorm.shape[:2]
        cx, cy = w // 2, h // 2
        center = np.array([cx, cy], dtype=np.float64)

        cnt, edges, dbg = self.findNeedleContour(imgNorm, debug=debugDraw)
        if cnt is None:
            print("Контур стрелки не найден.")
            return None, imgNorm

        mean, vals, vecs = self.principalAxesFromContour(cnt)
        axis = vecs[:, 0]
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        p1, p2, _ = self.contourEndpointsAlongAxis(cnt, mean, axis)

        d1 = np.linalg.norm(p1 - center)
        d2 = np.linalg.norm(p2 - center)
        if d1 < d2:
            base = p1
            tip = p2
        else:
            base = p2
            tip = p1

        vx = float(tip[0] - center[0])
        vy = float(tip[1] - center[1])
        thetaNeedle360 = self.toAngle360(vx, vy)

        imgVisual = imgNorm.copy()
        if debugDraw:
            cv2.drawContours(imgVisual, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(imgVisual, (cx, cy), 4, (0, 255, 0), -1)

            bx, by = int(round(base[0])), int(round(base[1]))
            tx, ty = int(round(tip[0])), int(round(tip[1]))
            cv2.circle(imgVisual, (bx, by), 4, (255, 0, 0), -1)
            cv2.circle(imgVisual, (tx, ty), 4, (0, 0, 255), -1)

            cv2.line(imgVisual, (cx, cy), (tx, ty), (0, 255, 255), 2)

            text = f"{thetaNeedle360:.1f} deg"
            cv2.putText(imgVisual, text, (tx + 5, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        print(f"Угол стрелки (0–360): {thetaNeedle360:.2f}°")
        return thetaNeedle360, imgVisual

    def computeTachValueFromAngles(self, thetaMin360, thetaNeedle360, maxValue):
        thetaMax360 = (180.0 - thetaMin360) % 360.0

        arcTotal = 360 - ((thetaMin360 - thetaMax360 + 360.0) % 360.0)
        arcNeedle = thetaNeedle360 - thetaMin360

        print(f"theta_min = {thetaMin360:.2f}°, theta_max = {thetaMax360:.2f}°")
        print(f"arc_total  (min->max по часовой)    = {arcTotal:.2f}°")
        print(f"arc_needle (min->needle по часовой) = {arcNeedle:.2f}°")

        if arcTotal < 1e-3:
            raise ValueError("Слишком малая дуга шкалы – проверь калибровку.")

        if arcNeedle > arcTotal:
            print("Стрелка вне рабочей дуги (по часовой от min до max)")
            return 0.0

        ratio = arcNeedle / arcTotal
        value = maxValue * ratio
        return value if value > 0 else 0.0

    def collectDetectionsInRegion(self, r, region=None, maxItems=8):
        lines = []
        if r.boxes is None or len(r.boxes) == 0:
            return lines

        rx1 = ry1 = rx2 = ry2 = None
        if region is not None:
            rx1, ry1, rx2, ry2 = region

        for box in r.boxes:
            clsId = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            if region is not None:
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue

            name = self.classNames.get(clsId, str(clsId))
            lines.append(f"{name}: {conf:.2f}")
            if len(lines) >= maxItems:
                break

        return lines

    def makeInfoPanel(self, value, detFull, detTach, height, inferTime=None, needleTime=None, frameExtractTime=None):
        panelWidth = 520
        panel = np.zeros((height, panelWidth, 3), dtype=np.uint8)

        x = 10
        y = 25
        lineH = 18
        colW = 250

        def putLine(text, color=(0, 255, 255), bold=False):
            nonlocal x, y
            if y > height - 12:
                x += colW
                y = 25
            if x > panelWidth - colW:
                return
            thickness = 2 if bold else 1
            cv2.putText(panel, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, thickness, cv2.LINE_AA)
            y += lineH

        putLine("Tachometer value:", bold=True)
        putLine(f"{value:.2f}", color=(0, 255, 0), bold=True)
        y += lineH // 2

        putLine("Timing:", bold=True)
        if inferTime is not None:
            putLine(f"YOLO: {inferTime * 1000:.1f} ms", color=(255, 255, 255))
        if needleTime is not None:
            putLine(f"Needle ({self.needleMethod}): {needleTime * 1000:.1f} ms", color=(255, 255, 255))
        if frameExtractTime is not None:
            putLine(f"Frame extract: {frameExtractTime * 1000:.1f} ms", color=(255, 255, 255))
        y += lineH // 2

        putLine("Detections (frame):", bold=True)
        if detFull:
            for t in detFull:
                putLine(f"- {t}")
        else:
            putLine("- none")
        y += lineH // 2

        return panel

    def sampleFramesFromVideo(self, path, periodSec):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("[ERROR] Не удалось открыть видео.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        frameInterval = max(int(round(fps * periodSec)), 1)

        frames = []
        frameIdx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frameIdx % frameInterval == 0:
                tSec = frameIdx / fps
                frames.append((frame.copy(), tSec, frameIdx))
            frameIdx += 1

        cap.release()
        print(f"Извлечено {len(frames)} кадров (каждые ~{periodSec} сек)\n")
        return frames

    def extractTachBoxes(self, r):
        tachBoxes = []
        if r.boxes is None or len(r.boxes) == 0:
            return tachBoxes

        for box in r.boxes:
            clsId = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if clsId == self.tachId:
                tachBoxes.append((conf, (int(x1), int(y1), int(x2), int(y2))))

        return tachBoxes

    def pickBestBox(self, boxes, hw):
        if not boxes:
            return None, None

        boxes.sort(key=lambda x: x[0], reverse=True)
        bestConf, (x1, y1, x2, y2) = boxes[0]

        h, w = hw
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return bestConf, None

        return bestConf, (x1, y1, x2, y2)


def main():
    mode = input("На чем будем тестировать? (Enter для стандартного видео): ").strip()

    if not mode:
        mode = "video"
        path = STANDARD_VIDEO_PATH
    elif mode.strip().lower() == "video":
        mode = "video"
        path = input("Введите путь к видео: ").strip()
    else:
        mode = "img"
        path = input("Введите путь к фото: ").strip()

    app = App(YOLO_WEIGHTS_PATH, mode, path)
    app.start()


if __name__ == "__main__":
    main()
