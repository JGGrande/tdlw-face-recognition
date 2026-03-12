import cv2
import os
import threading
import time
import numpy as np
from deepface import DeepFace
from typing import List, Dict, Any, Union

emotion_analysis_delay = 0.01  # Segundos entre cada análise de emoção
current_emotion = 'neutral'
current_face_region = None
frame_for_analysis = None
program_is_running = True

face_square_color = (0, 0, 255) # Vermelho
face_square_thickness = 2

def load_photos(directory: str) -> dict:
    """
    Carrega as fotos das emoções e as armazena em um dicionário.
    As chaves são os nomes das emoções do DeepFace.
    """
    emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral', 'fear', 'disgust']
    fotos_dict = {}
    
    for emotion in emotions:
        photo_path = os.path.join(directory, f"{emotion}.jpg")
        if os.path.exists(photo_path):
            img = cv2.imread(photo_path)

            if img is not None:
                fotos_dict[emotion] = img
            else:
                print(f"Erro: O arquivo {photo_path} existe, mas não pôde ser lido pelo OpenCV. Pode estar corrompido ou com extensão incorreta.")
                fotos_dict[emotion] = np.zeros((480, 480, 3), dtype=np.uint8)
        else:
            print(f"Aviso: Foto não encontrada para a emoção '{emotion}'. Crie o arquivo {photo_path}.")
            # Cria uma imagem preta de fallback caso falte alguma foto
            fotos_dict[emotion] = np.zeros((480, 480, 3), dtype=np.uint8)
            
    return fotos_dict

def get_first_emotion(result: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]) -> Dict[str, Any]:
    """
    A função get_first_emotion é usada para lidar com a estrutura de dados retornada pelo DeepFace.
    O resultado pode ser um vetor de detecções (se houver mais de um rosto) ou uma única lista de dicionários (se houver apenas um rosto).
    Esta função garante que sempre retornemos um dicionário com as emoções, mesmo que haja apenas um rosto.
    """
    if isinstance(result[0], list):
        # Se for uma lista, pegamos o primeiro elemento
        return result[0][0]
    return result[0]

def process_emotion_analysis():
    global current_emotion, current_face_region, frame_for_analysis, program_is_running
    
    while program_is_running:
        if frame_for_analysis is not None:
            try:
                emotion_detection_result = DeepFace.analyze(
                    frame_for_analysis, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    silent=True
                )

                first_face_result = get_first_emotion(emotion_detection_result)

                current_emotion = first_face_result['dominant_emotion']
                current_face_region = first_face_result['region']
            except Exception as e:
                current_face_region = None # Se não achar rosto, remove o quadrado
        time.sleep(emotion_analysis_delay)

def main():
    global frame_for_analysis, program_is_running, current_emotion

    # 1. Carrega o banco de imagens de emoções
    photos = load_photos("photos")
    
    # 2. Inicia a webcam
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    process_emotion_thread = threading.Thread(target=process_emotion_analysis, daemon=True, name="EmotionAnalysis")
    process_emotion_thread.start()

    print("Câmera iniciada. Pressione 'q' para sair.")
    
    current_emotion = 'neutral' # Emoção padrão inicial

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Espelha o frame para ficar mais natural (como um espelho)
        frame = cv2.flip(frame, 1)

        # 3. Envia o frame atual para a thread de análise
        frame_for_analysis = frame.copy()

        # 4. Desenha o quadrado e a emoção usando o resultado mais recente da thread
        if current_face_region is not None:
            x, y, w, h = current_face_region['x'], current_face_region['y'], current_face_region['w'], current_face_region['h']

            cv2.rectangle(
                frame,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=face_square_color,
                thickness=face_square_thickness
            )

            label = f"Emotion: {current_emotion}"
            label_y = y - 10 if y - 10 > 10 else y + h + 25

            cv2.putText(
                frame,
                text=label,
                org=(x, label_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=face_square_color,
                thickness=2
            )

        # 5. Pega a foto da emoção correspondente
        img_emotion = photos.get(current_emotion, photos['neutral'])
        
        # Redimensiona a foto da emoção para ter a mesma altura da webcam
        h_frame, w_frame = frame.shape[:2]
        h_emotion, w_emotion = img_emotion.shape[:2]
        
        # Calcula a nova largura mantendo a proporção da imagem
        new_width = int(w_emotion * (h_frame / h_emotion))
        resized_emotion_image = cv2.resize(img_emotion, (new_width, h_frame))

        # 5. Junta as duas imagens lado a lado
        final_screen = np.hstack((frame, resized_emotion_image))

        # 6. Exibe o resultado
        cv2.imshow('Qual a sua Emoção Interior?', final_screen)

        # Se pressionar a tecla 'q', fecha o programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    program_is_running = False

    process_emotion_thread.join()  # Aguarda a thread de análise de emoção terminar

    # Libera a câmera e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()