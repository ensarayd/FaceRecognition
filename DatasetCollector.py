import cv2
import os
import tkinter as tk


root = tk.Tk()
root.title("Kullanıcı ID Girişi")
root.geometry("800x600")


face_id = tk.StringVar()

label = tk.Label(root, text="Girilen Sayı: ", font=("Helvetica", 14)) # Kullanıcı ID'sini ekranda gösteren etiket

label.pack(pady=10)

def update_id(number):    # Kullanıcı ID'sini alacak Fonskiyon
    current = face_id.get()
    if len(current) < 2:
        face_id.set(current + str(number))
    label.config(text="Girilen Sayı: " + face_id.get())

def start_detection():
    if face_id.get():
        root.destroy()
    else:
        label.config(text="Lütfen bir sayı girin!")


for i in range(10):
    btn = tk.Button(root, text=str(i), width=5, height=2, command=lambda i=i: update_id(i))
    btn.pack(side=tk.LEFT, padx=5, pady=5)

start_button = tk.Button(root, text="Başlat", width=10, height=2, command=start_detection)
start_button.pack(pady=10)
root.mainloop()

if face_id.get():
    print("\n Yüzünüz analiz ediliyor lütfen bekleyiniz ...") # Kullanıcı ID'si Girildiyse yüz tanıma işlemine başla
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            
            cv2.imwrite("dataset/user." + str(face_id.get()) + '.' +  
                        str(count) + ".jpg", gray[y:y+h, x:x+w])
            
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 400: 
            break

    print("\n [INFO] Programdan Çıkılıyor")
    cam.release()
    cv2.destroyAllWindows()

else:
    print("Kullanıcı id girilmedi, programdan çıkılıyor.")
