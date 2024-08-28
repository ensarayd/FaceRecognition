import cv2
import numpy as np
import os 
import time
from PIL import Image, ImageDraw, ImageFont
import requests
import mysql.connector
from mysql.connector import Error



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['Taınınmıyor', 'Ensar',] 
cam = cv2.VideoCapture(0)
cam.set(3, 1280) 
cam.set(4, 720) 
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
save_path = 'tespit'
font_cv2 = cv2.FONT_HERSHEY_SIMPLEX
#post_url=''
photo_counter = 0
start_time=time.time()
last_id=None


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
            print("Kayıt Başarıyla veri tabanına kaydoldu")
    except Error as e:
        print(f"Error: {e}")
        print("Bağlantı veya SQL Query hatası")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1) # 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))

            if id != last_id or time.time() - start_time >= 20:
                photo_counter += 1
                face_img = img[y:y + h, x:x + w]
                date_text = time.strftime("%Y-%m-%d_%H-%M-%S")

                face_filename = f"{save_path}/User_{id}_{date_text}.jpg"
                face_username=f"{id}"
                cv2.imwrite(face_filename, face_img)
                
                print(f"[INFO] Kaydedildi {face_filename}")
                print(face_username)
                insert_to_database(face_username)
                
                last_id = id  
                start_time = time.time()  
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font_cv2, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font_cv2, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    cv2.imshow('Tespit',img) 
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

print("\n [INFO] Programdan Çıkılıyor")
cam.release()
cv2.destroyAllWindows()