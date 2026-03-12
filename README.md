# TDLW Face Recognition

Repositório com 4 projetos independentes de visão computacional com Python, OpenCV e DeepFace:

- **2d-face-recognition**: detecção de rostos em tempo real usando Haar Cascade.
- **age-detector**: estimativa de idade em tempo real com DeepFace.
- **face-verification**: verificação facial comparando o rosto da câmera com uma imagem de referência (`img/user.png`).
- **facial-expression**: reconhecimento de emoção em tempo real e exibição de imagem correspondente da pasta `photos/`.

---

## Pré-requisitos

- **Windows** (testado para uso com webcam local)
- **Python 3.10+**
- **pip** atualizado
- Webcam conectada e liberada para uso

> Dica: execute cada projeto em um ambiente virtual próprio para evitar conflitos de versão entre dependências.

---

## Estrutura do projeto

```txt
2d-face-recognition/
  main.py
  requirements.txt
age-detector/
  main.py
  requirements.txt
face-verification/
  main.py
  requirements.txt
  img/
    user.png
facial-expression/
  main.py
  requirements.txt
  photos/
    angry.jpg
    disgust.jpg
    fear.jpg
    happy.jpg
    neutral.jpg
    sad.jpg
    surprise.jpg
```

---

## Como rodar cada projeto

Os comandos abaixo estão em **PowerShell**.

### 1) 2d-face-recognition

```powershell
cd 2d-face-recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

**O que faz:**
- Abre a webcam.
- Detecta rostos com Haar Cascade.
- Desenha retângulo vermelho e contador de rostos.
- Fecha ao pressionar `q`.

---

### 2) age-detector

```powershell
cd age-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

**O que faz:**
- Detecta rostos em cada frame com OpenCV.
- Estima idade aproximada com `DeepFace.analyze` (ação `age`) a cada alguns frames para melhorar desempenho.
- Exibe idade abaixo do retângulo do rosto.
- Fecha ao pressionar `q`.

---

### 3) face-verification

```powershell
cd face-verification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

**O que faz:**
- Usa `img/user.png` como referência de usuário autorizado.
- Detecta rosto na webcam e compara com a referência via `DeepFace.verify`.
- Mostra **Autorizado** (com preview da imagem de referência) ou **Nao autorizado**.
- Fecha ao pressionar `q`.

**Importante:**
- Garanta que `face-verification/img/user.png` exista e tenha uma foto nítida do usuário autorizado.

---

### 4) facial-expression

```powershell
cd facial-expression
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

**O que faz:**
- Analisa a emoção dominante em tempo real com DeepFace (thread separada para reduzir travamentos).
- Desenha o retângulo vermelho no rosto e exibe o nome da emoção.
- Mostra, lado a lado com a webcam, a imagem correspondente da pasta `photos/`.
- Fecha ao pressionar `q`.

**Importante:**
- As imagens de emoção devem existir em `facial-expression/photos` com estes nomes:
  - `happy.jpg`, `sad.jpg`, `angry.jpg`, `surprise.jpg`, `neutral.jpg`, `fear.jpg`, `disgust.jpg`
- O código usa `cv2.VideoCapture(1)`. Se não abrir câmera, altere para `0` no `main.py`.

---

## Problemas comuns

- **Erro ao abrir câmera**
  - Feche outros apps que estejam usando webcam (Teams, Zoom, navegador, etc.).
  - Teste outro índice de câmera (`0`, `1`, `2`) no `cv2.VideoCapture(...)`.

- **Instalação do DeepFace/TensorFlow lenta ou falhando**
  - Atualize pip: `python -m pip install --upgrade pip`
  - Use ambiente virtual novo.
  - Confirme que a versão do Python é compatível com as dependências.

- **Desempenho baixo (FPS baixo)**
  - Reduza resolução da câmera no código.
  - Aumente o intervalo de análise (por exemplo, `ANALYZE_INTERVAL` no age-detector).

---

## Observações

- Cada pasta é um projeto independente com seu próprio `requirements.txt`.
- Você pode manter um ambiente virtual por pasta para isolamento de dependências.
- Para sair das aplicações em execução, pressione `q` na janela do OpenCV.
