import cv2
from deepface import DeepFace

# Caminho da imagem de referência
IMAGEM_REFERENCIA = "img/user.png"

# Preview do usuário autorizado para exibir na tela
preview_usuario = cv2.imread(IMAGEM_REFERENCIA)
preview_usuario = cv2.resize(preview_usuario, (100, 100))

# Detector de rostos do OpenCV
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0)

resultado_autorizado = None
contador = 0
INTERVALO_VERIFICACAO = 20  # Executa o DeepFace a cada 20 frames para melhor desempenho

while True:
    ret, frame = camera.read()
    if not ret:
        break

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos = detector.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in rostos:
        # Desenha retângulo vermelho ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Compara o rosto com a imagem de referência a cada INTERVALO_VERIFICACAO frames
        if contador % INTERVALO_VERIFICACAO == 0:
            try:
                rosto_recortado = frame[y:y + h, x:x + w]
                verificacao = DeepFace.verify(
                    rosto_recortado,
                    IMAGEM_REFERENCIA,
                    enforce_detection=False
                )
                resultado_autorizado = verificacao["verified"]
            except Exception:
                resultado_autorizado = False

        if resultado_autorizado:
            # Exibe o preview do usuário autorizado abaixo do retângulo
            y1 = y + h + 5
            y2 = y1 + preview_usuario.shape[0]
            x1 = x
            x2 = x1 + preview_usuario.shape[1]
            if y2 < frame.shape[0] and x2 < frame.shape[1]:
                frame[y1:y2, x1:x2] = preview_usuario
        elif resultado_autorizado is not None:
            cv2.putText(
                frame, "Nao autorizado",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

    contador += 1
    cv2.imshow("Verificacao Facial", frame)

    # Pressione 'q' para encerrar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
