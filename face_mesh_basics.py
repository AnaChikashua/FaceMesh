import cv2
import mediapipe as mp
import time
import pafy

url = "https://youtu.be/Xygk7UjKM2g?si=kRZNi31OZbeq-Khu"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

cap = cv2.VideoCapture(best.url)
p_time = 0

mp_draw = mp.solutions.drawing_utils
mp_face_mash = mp.solutions.face_mesh
face_mash = mp_face_mash.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2)
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mash.process(img_rgb)
    if results.multi_face_landmarks:
        for face_lm in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_lm, mp_face_mash.FACEMESH_CONTOURS, draw_spec, draw_spec)
            for lm in face_lm.landmark:
                ih, iw, ic = img.shape
                x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z * ic)
                print(x, y, z)
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f"FPS: {str(int(fps))}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("image", img)

    cv2.waitKey(1)
