import cv2

# ─────────────────────────────────────────────
# Reconhecimento Facial com OpenCV
# ─────────────────────────────────────────────
# Utiliza o classificador Haar Cascade, que já
# vem embutido no pacote opencv-python.
# Nenhum arquivo externo precisa ser baixado.
# ─────────────────────────────────────────────


def carregar_classificador() -> cv2.CascadeClassifier:
    caminho = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    classificador = cv2.CascadeClassifier(caminho)

    if classificador.empty():
        raise RuntimeError("Erro: classificador Haar Cascade não pôde ser carregado.")

    return classificador


def detectar_rostos(frame_cinza: cv2.Mat, classificador: cv2.CascadeClassifier):
    rostos = classificador.detectMultiScale(
        frame_cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return rostos


def desenhar_rostos(frame: cv2.Mat, rostos: list) -> int:
    # Cor vermelha no formato BGR do OpenCV
    COR_VERMELHO = (0, 0, 255)
    ESPESSURA_BORDA = 2

    for (x, y, largura, altura) in rostos:
        cv2.rectangle(
            frame,
            (x, y),                        # canto superior esquerdo
            (x + largura, y + altura),     # canto inferior direito
            COR_VERMELHO,
            ESPESSURA_BORDA,
        )

        cv2.putText(
            frame,
            text="Rosto detectado",
            org=(x, y - 10),               # 10px acima do retângulo
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=COR_VERMELHO,
            thickness=2,
        )

    return len(rostos)


def main():
    classificador = carregar_classificador()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível acessar a câmera.")
        return

    print("Câmera iniciada. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro: falha ao capturar frame da câmera.")
            break

        frame = cv2.flip(frame, 1)

        frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostos = detectar_rostos(frame_cinza, classificador)

        quantidade = desenhar_rostos(frame, rostos)

        cv2.putText(
            frame,
            text=f"Rostos: {quantidade}",
            org=(15, 35),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.9,
            color=(0, 255, 0),   # verde
            thickness=2,
        )

        cv2.imshow("Reconhecimento Facial", frame)

        # 'Sai ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Câmera encerrada.")


if __name__ == "__main__":
    main()
