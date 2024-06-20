import cv2
import numpy as np
from time import time
import datetime
import torch
from ultralytics import YOLO
import cvzone
import math
import matplotlib.pyplot as plt

# Kalman Filtresi Parametreleri
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

cap = cv2.VideoCapture("yenivid.mp4")

start_time = 0
end_time = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device: ", device)

model = YOLO("egitim_uav/v8s_359epoch/best.pt").to(device)
classNames = ["-1", "0"]
classNames[1] = "uav"
classNames[0] = "uav"

# Hata ölçümünü saklamak için listeler
true_positions = []
predicted_positions = []
errors = []

# Başlangıç pozisyonunu belirleme
initial_x1 = initial_y1 = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1024, 768))

    x = frame.shape[1]
    y = frame.shape[0]

    end_time = time()

    x1, y1 = int(x * 0.25), int(y * 0.1)
    x2, y2 = int(x * 0.75), int(y * 0.9)

    # Başlangıç pozisyonunu kaydetme
    if initial_x1 is None or initial_y1 is None:
        initial_x1, initial_y1 = x1, y1
        kalman.statePre = np.array([[initial_x1], [initial_y1], [0], [0]], np.float32)
        kalman.statePost = np.array([[initial_x1], [initial_y1], [0], [0]], np.float32)

    cv2.rectangle(frame, (0, 0), (x, y), (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.putText(frame, "Ak: Kamera Gorus Alani", (10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Av: Hedef Vurus Alani", (int(x / 4) + 5, int(9 * y / 10) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    center_x = int(frame.shape[1] / 2)
    center_y = int(frame.shape[0] / 2)
    crosshair_size = 2
    crosshair_tickness = 1

    cv2.line(frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (0, 255, 0),
             crosshair_tickness)
    cv2.line(frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (0, 255, 0),
             crosshair_tickness)

    results = model(frame, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                w, h = abs(x2 - x1), abs(y2 - y1)

                center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
                obj_center = (int(x1 + w / 2), int(y1 + h / 2))

                # Gerçek konumu ve tahmin edilen konumu saklama
                true_positions.append(obj_center)

                # Kalman Filtresini Güncelleme
                prediction = kalman.predict()
                kalman.correct(np.array([[obj_center[0]], [obj_center[1]]], np.float32))

                # Tahmin Edilen Pozisyonu Çiz
                predicted_center = (int(prediction[0][0]), int(prediction[1][0]))
                predicted_positions.append(predicted_center)

                # Hata ölçümünü hesaplama (Öklidyen Mesafe)
                error = math.sqrt((predicted_center[0] - obj_center[0])**2 + (predicted_center[1] - obj_center[1])**2)
                errors.append(error)

                cv2.line(frame, center, predicted_center, (255, 255, 255), 1)

            if int(frame.shape[1] / 4) < x1 < int(3 * frame.shape[1] / 4) and int(frame.shape[0] / 10) < y1 < int(
                    9 * frame.shape[0] / 10):
                inside = True
            else:
                inside = False

            if inside:
                cv2.putText(frame, "Inside", (int(frame.shape[1] / 4) + 435, int(9 * frame.shape[0] / 10) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 200), 2, cv2.LINE_AA, False)
            else:
                cv2.putText(frame, "Outside", (int(frame.shape[1] / 4) + 435, int(9 * frame.shape[0] / 10) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 200), 2, cv2.LINE_AA, False)

        fps = 1 / (end_time - start_time)
        fps = int(fps)

        start_time = end_time

        t = datetime.datetime.now()
        server_saati = str(t.hour)
        server_dakika = str(t.minute)
        server_saniye = str(t.second)
        server_mili = str(t.microsecond / 1000)
        saat = {"saat": server_saati, "dakika": server_dakika, "saniye": server_saniye, "milisaniye": server_mili}

        cv2.putText(frame, (f"fps:{str(fps)}"), (20, 20), 1, 1, (255, 255, 255), 2, cv2.FONT_HERSHEY_DUPLEX)

        cv2.putText(frame, str(saat), (450, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 128, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Hata oranlarını ve pozisyonları görselleştirme
times = np.arange(len(errors))

plt.figure(figsize=(12, 6))

# Gerçek ve Tahmin Edilen X Pozisyonları
plt.subplot(2, 1, 1)
plt.plot(times, [pos[0] for pos in true_positions], label="Gerçek X")
plt.plot(times, [pos[0] for pos in predicted_positions], label="Tahmin X")
plt.xlabel("Frame")
plt.ylabel("X Pozisyonu")
plt.legend()

# Gerçek ve Tahmin Edilen Y Pozisyonları
plt.subplot(2, 1, 2)
plt.plot(times, [pos[1] for pos in true_positions], label="Gerçek Y")
plt.plot(times, [pos[1] for pos in predicted_positions], label="Tahmin Y")
plt.xlabel("Frame")
plt.ylabel("Y Pozisyonu")
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(times, errors, label="Hata Oranı")
plt.xlabel("Frame")
plt.ylabel("Hata Oranı (Öklidyen Mesafe)")
plt.legend()

plt.show()
