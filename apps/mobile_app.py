from fasthtml.common import *
import base64, cv2, numpy as np, time, os, sys, mediapipe as mp, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import ModelLoader
from core.processor import FrameProcessor

hdrs = (picolink, Meta(name="viewport", content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"))
app, rt = fast_app(hdrs=hdrs)

loader = ModelLoader(models_dir="../models")
processor = FrameProcessor()

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
    status.innerHTML = "Permissão...";
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
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
                let tmp = document.createElement('canvas');
                tmp.width = video.videoWidth; tmp.height = video.videoHeight;
                tmp.getContext('2d').drawImage(video, 0, 0);
                ws.send(tmp.toDataURL('image/jpeg', 0.5));
            }
        }, 100);
    };
    ws.onmessage = (e) => {
        let res = JSON.parse(e.data);
        let img = new Image();
        img.onload = () => {
            canvas.width = window.innerWidth * 0.95;
            canvas.height = (img.height / img.width) * canvas.width;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            res.detections.forEach(d => {
                ctx.fillStyle = "#00FF00"; ctx.font = "bold 18px Arial";
                let s = canvas.width / img.width;
                ctx.fillText(d.side + ": " + d.prediction, d.x * s, d.y * s + 25);
            });
        };
        img.src = res.image;
    };
}
"""

@rt("/")
def get():
    return Title("FrontMachine Mobile"), Main(
        Div(H2("Reconhecimento de Gestos"), 
            Button("🚀 Ativar Câmera", onclick="startApp()", style="padding: 15px 30px;"),
            id="overlay", style="text-align: center; padding: 40px;"),
        P(id="status", style="text-align: center;"),
        Video(id="webcam", autoplay=True, playsinline=True, muted=True, style="display: none;"),
        Canvas(id="output", style="width: 100%; border-radius: 8px;"),
        Script(JS_CODE)
    )

@app.ws("/ws")
async def ws(send, receive):
    global last_ts
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
                        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        await send({'type': 'websocket.send', 'text': json.dumps({"image": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}", "detections": processor.get_detections(frame, res, loader.clf)})})
            except: break

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    serve(port=port)
