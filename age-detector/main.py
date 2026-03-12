import cv2
import numpy as np
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

ANALYZE_INTERVAL: int = 15  # Analisa idade a cada N frames para melhor desempenho


def estimate_age(frame: np.ndarray) -> str:
    try:
        result = DeepFace.analyze(frame, actions=["age"], enforce_detection=False)
        result = result[0] if isinstance(result, list) else result
        return f"Idade: ~{result['age']} anos"
    except Exception:
        return ""


def draw_face(frame: np.ndarray, x: int, y: int, w: int, h: int, age_text: str) -> None:
    # Quadrado vermelho ao redor do rosto
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Texto com a idade abaixo do quadrado
    if age_text:
        cv2.putText(
            frame,
            text=age_text, 
            org=(x, y + h + 28),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.8, 
            color=(0, 0, 255), 
            thickness=2
        )


def run() -> None:
    cap = cv2.VideoCapture(0)
    age_text: str = ""
    frame_count: int = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Estima a idade periodicamente para não travar o vídeo
        if frame_count % ANALYZE_INTERVAL == 0 and len(faces) > 0:
            age_text = estimate_age(frame)

        for (x, y, w, h) in faces:
            draw_face(frame, x, y, w, h, age_text)

        frame_count += 1
        cv2.imshow("Detector de Idade", frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
