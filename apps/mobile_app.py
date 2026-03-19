from fasthtml.common import *
import base64, cv2, numpy as np, time, os, sys, mediapipe as mp, json

# Setup core - Caminho ajustado para um nível acima
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import ModelLoader
from core.processor import FrameProcessor

# Configuração para Mobile
hdrs = (picolink, Meta(name="viewport", content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"))
app, rt = fast_app(hdrs=hdrs)

# Caminho dos modelos ajustado
loader = ModelLoader(models_dir="../models")
processor = FrameProcessor()

# Controle de tempo global
last_ts = 0

JS_CODE = """
let video = document.getElementById('webcam');
let canvas = document.getElementById('output');
let status = document.getElementById('status');
let overlay = document.getElementById('overlay');
let ctx = canvas.getContext('2d');
let ws;

function startApp() {
    ws = new WebSocket((window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws');
    status.innerHTML = "Solicitando permissão...";
    navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, 
        audio: false 
    })
    .then(stream => { 
        video.srcObject = stream;
        video.play();
        overlay.style.display = 'none';
        status.innerHTML = "Processando...";
        setupLoop();
    })
    .catch(err => {
        status.innerHTML = "ERRO: " + err;
    });
}

function setupLoop() {
    video.onplay = () => {
        setInterval(() => {
            if (ws && ws.readyState === 1 && video.readyState === 4) {
                let tempCanvas = document.createElement('canvas');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                tempCanvas.getContext('2d').drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
                ws.send(tempCanvas.toDataURL('image/jpeg', 0.5));
            }
        }, 100);
    };

    ws.onmessage = (event) => {
        let response = JSON.parse(event.data);
        let img = new Image();
        img.onload = () => {
            canvas.width = window.innerWidth * 0.95;
            canvas.height = (img.height / img.width) * canvas.width;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            response.detections.forEach(det => {
                ctx.fillStyle = "#00FF00"; ctx.font = "bold 18px Arial";
                let scale = canvas.width / img.width;
                ctx.fillText(det.side + ": " + det.prediction, det.x * scale, det.y * scale + 25);
            });
        };
        img.src = response.image;
    };
}
"""

@rt("/")
def get():
    return Title("FrontMachine Mobile"), Main(
        Div(
            H2("Reconhecimento de Gestos"),
            Button("🚀 Ativar Câmera", onclick="startApp()", style="padding: 15px 30px; font-size: 1.1em;"),
            id="overlay", style="text-align: center; padding: 40px;"
        ),
        P(id="status", style="color: grey; text-align: center;"),
        Video(id="webcam", autoplay=True, playsinline=True, muted=True, style="display: none;"),
        Canvas(id="output", style="width: 100%; border: 2px solid #333; border-radius: 8px;"),
        Script(JS_CODE)
    )

@app.ws("/ws")
async def ws(send, receive):
    global last_ts
    HO, BO = mp.tasks.vision.HandLandmarkerOptions, mp.tasks.BaseOptions
    opts = HO(base_options=BO(model_asset_path=loader.hand_landmarker_path), 
              running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
    
    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as landmarker:
        while True:
            try:
                msg = await receive()
                if msg['type'] == 'websocket.receive':
                    _, encoded = msg['text'].split(",", 1)
                    frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), 1)
                    if frame is not None:
                        frame = cv2.flip(frame, 1)
                        ts = int(time.time() * 1000)
                        if ts <= last_ts: ts = last_ts + 1
                        last_ts = ts
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        result = landmarker.detect_for_video(mp_image, ts)
                        detections = processor.get_detections(frame, result, loader.clf)
                        if result.hand_landmarks:
                            for hand_lms in result.hand_landmarks: processor.draw_hand_landmarks(frame, hand_lms)
                        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        await send({'type': 'websocket.send', 'text': json.dumps({
                            "image": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}",
                            "detections": detections
                        })})
            except Exception as e:
                print(f"Erro WS: {e}")
                break

if __name__ == "__main__":
    serve(port=8001)
