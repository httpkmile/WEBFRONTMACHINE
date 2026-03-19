from fasthtml.common import *
import base64, cv2, numpy as np, time, os, sys, mediapipe as mp, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import ModelLoader
from core.processor import FrameProcessor

hdrs = (picolink, Meta(name="viewport", content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"))
app, rt = fast_app(hdrs=hdrs)

loader = ModelLoader()
processor = FrameProcessor()
last_ts = 0

JS_CODE = """
let video, canvas, status, logger, ws, ctx, sendInterval;

function log(msg, color="#333") {
    let div = document.createElement('div');
    div.style.color = color;
    div.style.fontSize = "0.7em";
    div.innerHTML = "[" + new Date().toLocaleTimeString() + "] " + msg;
    logger.prepend(div);
}

function startApp() {
    video = document.getElementById('webcam');
    canvas = document.getElementById('output');
    status = document.getElementById('status');
    logger = document.getElementById('logger');
    ctx = canvas.getContext('2d');
    
    let protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    let url = protocol + window.location.host + '/ws';
    log("Conectando ao WS: " + url, "blue");
    
    ws = new WebSocket(url);

    ws.onopen = () => {
        log("WebSocket Conectado!", "green");
        log("Solicitando Câmera...");
        navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: {ideal: 640}, height: {ideal: 480} }, 
            audio: false 
        })
        .then(stream => { 
            video.srcObject = stream;
            video.play();
            document.getElementById('overlay').style.display = 'none';
            log("Câmera Liberada!", "green");
            setupLoop();
        })
        .catch(err => {
            log("ERRO Câmera: " + err.name + ": " + err.message, "red");
        });
    };

    ws.onerror = (e) => log("ERRO WebSocket: Falha na conexão.", "red");
    ws.onclose = () => log("Conexão encerrada pelo servidor.", "orange");
}

function setupLoop() {
    let tempCanvas = document.createElement('canvas');
    let tempCtx = tempCanvas.getContext('2d');
    
    sendInterval = setInterval(() => {
        if (ws && ws.readyState === 1 && video.readyState === 4) {
            tempCanvas.width = video.videoWidth;
            tempCanvas.height = video.videoHeight;
            tempCtx.drawImage(video, 0, 0);
            ws.send(tempCanvas.toDataURL('image/jpeg', 0.5));
        }
    }, 100);

    ws.onmessage = (e) => {
        let res = JSON.parse(e.data);
        let img = new Image();
        img.onload = () => {
            canvas.width = window.innerWidth * 0.95;
            canvas.height = (img.height / img.width) * canvas.width;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            res.detections.forEach(d => {
                ctx.fillStyle = "#00FF00"; ctx.font = "bold 16px Arial";
                let s = canvas.width / (img.width || 1);
                ctx.fillText(d.side + ": " + d.prediction, d.x * s, d.y * s + 20);
            });
        };
        img.src = res.image;
    };
}

window.onerror = (msg, url, line) => log("JS ERROR: " + msg + " (L" + line + ")", "red");
"""

@rt("/")
def get():
    return Title("Debugger Vision"), Main(
        Div(H2("Reconhecimento de Gestos"), 
            P("Toque abaixo para iniciar o diagnóstico avançado:"),
            Button("🚀 Iniciar App", onclick="startApp()", style="padding: 15px; width: 100%;"),
            id="overlay", style="text-align: center; padding: 20px;"),
        P(id="status", style="text-align: center; font-size: 0.8em;"),
        Canvas(id="output", style="width: 100%; height: auto; border: 1px solid #ddd;"),
        Video(id="webcam", autoplay=True, playsinline=True, muted=True, style="display: none;"),
        Div(id="logger", style="height: 150px; overflow-y: auto; background: #eee; padding: 10px; border-top: 1px solid #ccc; font-family: monospace;"),
        Script(JS_CODE)
    )

@app.ws("/ws")
async def ws(send, receive):
    global last_ts
    print(">>> WS: Nova conexão iniciada em Railway")
    HO, BO = mp.tasks.vision.HandLandmarkerOptions, mp.tasks.BaseOptions
    opts = HO(base_options=BO(model_asset_path=loader.hand_landmarker_path), 
              running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as m:
        while True:
            try:
                msg = await receive()
                if msg['type'] == 'websocket.receive':
                    _, b64 = msg['text'].split(",", 1)
                    frame = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), 1)
                    if frame is not None:
                        frame = cv2.flip(frame, 1)
                        ts = int(time.time() * 1000)
                        if ts <= last_ts: ts = last_ts + 1
                        last_ts = ts
                        res = m.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
                        if res.hand_landmarks:
                            for hand_lms in res.hand_landmarks: processor.draw_hand_landmarks(frame, hand_lms)
                        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        await send({'type': 'websocket.send', 'text': json.dumps({"image": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}", "detections": processor.get_detections(frame, res, loader.clf)})})
            except Exception as e: 
                print(f"Erro no Loop WS: {e}")
                break

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    serve(port=port)
