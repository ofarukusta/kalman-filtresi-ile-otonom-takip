import cv2
import torch
from ultralytics import YOLO
import cvzone
import math


cap=cv2.VideoCapture("CokluHavaAraci.mp4")

device= "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device: ", device)

model= YOLO("egitim_uav/v8s_359epoch/best.pt").to(device)
classNames=["-1","0"]
classNames[1] = "uav"
classNames[0] = "uav"

while True: 

    ret,frame=cap.read()
    frame=cv2.resize(frame, (640,640))
    
 


    x = frame.shape[1]
    y = frame.shape[0]


  

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

                

  

            

    

    


    cv2.imshow('Tespit Deneme Arayuzu', frame)

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