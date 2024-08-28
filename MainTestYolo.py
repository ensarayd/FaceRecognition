import cv2
import numpy as np
import os
import time
import mysql.connector
from mysql.connector import Error
from ultralytics import YOLO


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

model = YOLO('Models/yolov8n-face.pt').to('cuda')   #cuda çekirdeği varsa cuda ile çalışacak


names = ['Tanınmıyor', 'Ensar']


cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480)  

save_path = 'Tespit'
if not os.path.exists(save_path):
    os.makedirs(save_path)

photo_counter = 0
start_time = time.time()
last_id = None


db_config = {
    'host': 'localhost',
    'database': 'mavrukdb',
    'user': 'elmanteams',
    'password': 'kD4422et'
}

def insert_to_database(user_id):
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            cursor = connection.cursor()
            sql_insert_query = """INSERT INTO face_detect (username) VALUES (%s)"""
            data = (user_id,)
            cursor.execute(sql_insert_query, data)
            connection.commit()
            print("Kayıt başarıyla veri tabanına kaydoldu")
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

while True:
    ret, img = cam.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

   
    start_frame_time = time.time()

 
    results = model(img, conf=0.5)  

    
    for result in results: 
        faces = result.boxes.xyxy  
        for face in faces:
            x1, y1, x2, y2 = [int(coord) for coord in face[:4]]


            face_img = img[y1:y2, x1:x2]
            video_filename = f"{save_path}/User_{photo_counter}.avi"
            out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 10, (x2 - x1, y2 - y1))

            start_video_time = time.time()
            while time.time() - start_video_time < 5:
                ret, img = cam.read()
                face_img = img[y1:y2, x1:x2]
                out.write(face_img)
            

            out.release()
            print(f"[INFO] Video kaydedildi: {video_filename}")
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            id, confidence = recognizer.predict(gray_face)

            if confidence < 100:
                id_name = names[id]
                confidence_text = f"  {round(100 - confidence)}%"
                if insert_to_database(id_name):
                    os.remove(video_filename)  
                    print(f"[INFO] Video silindi: {video_filename}")
                last_id = id
            else:
                id_name="unkownn"
                confidence_text = "  {0}%".format(round(100 - confidence))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(id_name), (x1 + 5, y1 - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence_text), (x1 + 5, y2 + 25), font, 1, (255, 255, 0), 1)

    end_frame_time = time.time()
    fps = 1 / (end_frame_time - start_frame_time)
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


    k = cv2.waitKey(10) & 0xff
    if k == 27:  
        break


print("\n [INFO] Programdan çıkılıyor")
cam.release()
cv2.destroyAllWindows()
