import cv2
import numpy as np
from time import time
import datetime
import torch
from ultralytics import YOLO
import cvzone
import math


cap=cv2.VideoCapture("yenivid.mp4")



start_time = 0
end_time = 0

device= "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device: ", device)

model= YOLO("egitim_uav/v8s_359epoch/best.pt").to(device)
classNames=["-1","0"]
classNames[1] = "uav"
classNames[0] = "uav"

while True: 

    ret,frame=cap.read()
    frame=cv2.resize(frame, (1024,768))
    
 


    x = frame.shape[1]
    y = frame.shape[0]



    x1, y1 = int(x * 0.25), int(y * 0.1)
    x2, y2 = int(x * 0.75), int(y * 0.9)

    cv2.rectangle(frame, (0, 0), (x, y), (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.putText(frame,"Ak:Kamera Gorus Alani",(10,int(y)-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame,"Av: Hedef Vurus Alani",(int(x/4)+5, int(9*y/10)-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    center_x = int(frame.shape[1] / 2)
    center_y = int(frame.shape[0] / 2)
    crosshair_size = 2
    crosshair_thickness = 1

    cv2.line(frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (0, 255, 0), crosshair_thickness)
    # Yatay çizgi
    cv2.line(frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (0, 255, 0), crosshair_thickness)

    #center_x = int(frame.shape[1] / 2)
    #center_y = int(frame.shape[0] / 2)

    results=model(frame,stream=True,verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0]*100)) /100
            cls = int(box.cls[0])
            if conf > 0.5:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cvzone.putTextRect(frame,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)), scale=1, thickness=1)
                w,h = abs(x2 - x1), abs(y2 - y1)

                

                

                center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
                obj_center = [int(x1 + w / 2), int(y1 + h / 2)]
                cv2.line(frame, center, obj_center, (255, 255, 255), 1)

        #    obj_center=[int(x1 + w / 2), int(y1 + h / 2)]
        #    cv2.line(frame, center, obj_center, (255, 255, 255), 1)



            

    cv2.imshow('Hava Araci Takip Arayuzu', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()

# Eğitim çıktısı üzerinden, uav kontrolü yapılıyor
# UAV görünürse kare içerisine alınıp takip vektörü çekiliyor
# FPS ve saat ekrana bastırılıyor


# Rakip Hava Araçlarının merkez X ve Y koordinatları çekilecek
# Öklid ile mesafesi ölçülecek
# Kalman ile nereye gidebileceği tahmin edilecek
# Kayıt alınarak demo oluşturulacak