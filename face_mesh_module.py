import cv2
import mediapipe as mp
import time
import pafy


class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mash = mp.solutions.face_mesh
        self.face_mash = self.mp_face_mash.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks,
                                                    self.min_detection_confidence, self.min_tracking_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def find_face_mesh(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mash.process(img_rgb)
        faces = []
        if self.results.multi_face_landmarks:
            for face_lm in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lm, self.mp_face_mash.FACEMESH_CONTOURS, self.draw_spec,
                                                self.draw_spec)
                face = []
                for _id, lm in enumerate(face_lm.landmark):
                    ih, iw, ic = img.shape
                    x, y, z = int(lm.x * iw), int(lm.y * ih), int(lm.z * ic)
                    cv2.putText(img, f"{_id}", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)

                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    url = "https://youtu.be/Xygk7UjKM2g?si=kRZNi31OZbeq-Khu"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")

    cap = cv2.VideoCapture(best.url)
    p_time = 0

    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img)
        if len(faces):
            print(len(faces))
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, f"FPS: {str(int(fps))}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
