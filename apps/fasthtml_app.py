from fasthtml.common import *
import base64, cv2, numpy as np, time, os, sys, mediapipe as mp, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import ModelLoader
from core.processor import FrameProcessor

app, rt = fast_app(hdrs=(picolink,))
loader = ModelLoader()
processor = FrameProcessor()

JS_CODE = """
let video = document.getElementById('webcam');
let canvas = document.getElementById('output');
let ctx = canvas.getContext('2d');
let gestureImg = document.getElementById('gesture_img');
let ws = new WebSocket((window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws');

navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(stream => { video.srcObject = stream; })
    .catch(err => { console.error("Erro webcam: ", err); });

video.addEventListener('play', () => {
    function sendFrame() {
        if (ws.readyState === WebSocket.OPEN) {
            let tempCanvas = document.createElement('canvas');
            tempCanvas.width = video.videoWidth; tempCanvas.height = video.videoHeight;
            tempCanvas.getContext('2d').drawImage(video, 0, 0);
            ws.send(tempCanvas.toDataURL('image/jpeg', 0.6));
        }
        setTimeout(sendFrame, 40); 
    }
    sendFrame();
});

ws.onmessage = (event) => {
    let response = JSON.parse(event.data);
    if (response.gesture_image) {
        gestureImg.src = `/assets/${response.gesture_image}.png`;
        gestureImg.style.display = 'block';
    } else {
        gestureImg.style.display = 'none';
    }
    let img = new Image();
    img.onload = () => {
        canvas.width = img.width; canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        response.detections.forEach(det => {
            ctx.fillStyle = "#00FF00"; ctx.font = "bold 20px Arial";
            ctx.fillText(`${det.side}: ${det.prediction}`, det.x - 50, det.y + 30);
            ctx.beginPath(); ctx.arc(det.x, det.y, 5, 0, 2 * Math.PI); ctx.fill();
        });
    };
    img.src = response.image;
};
"""

@rt("/assets/{fname:path}")
async def get(fname:str): return FileResponse(f"assets/{fname}")

@rt("/")
def get():
    return Title("FrontMachine - Desktop"), Main(
        H1("Deteção de Gestos"),
        Div(
            Div(Video(id="webcam", autoplay=True, style="display: none;"), 
                Canvas(id="output", style="width: 100%; border: 2px solid #333;"), style="flex: 2"),
            Div(H3("Gesto:"), Img(id="gesture_img", style="width: 200px; display: none;"), style="flex: 1; padding: 20px;"),
            style="display: flex;"
        ),
        Script(JS_CODE)
    )

@app.ws("/ws")
async def ws(send, receive):
    HO, BO = mp.tasks.vision.HandLandmarkerOptions, mp.tasks.BaseOptions
    opts = HO(base_options=BO(model_asset_path=loader.hand_landmarker_path), 
              running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as m:
        while True:
            try:
                msg = await receive()
                if msg['type'] == 'websocket.receive':
                    _, encoded = msg['text'].split(",", 1)
                    frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), 1)
                    if frame is not None:
                        frame = cv2.flip(frame, 1)
                        ts = int(time.time() * 1000)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        result = m.detect_for_video(mp_image, ts)
                        detections = processor.get_detections(frame, result, loader.clf)
                        gesture_image_name = detections[0]['prediction'] if (len(detections) > 0 and all(d['prediction'] == detections[0]['prediction'] for d in detections)) else None
                        clean_frame = frame.copy()
                        if result.hand_landmarks:
                            for hand_lms in result.hand_landmarks: processor.draw_hand_landmarks(clean_frame, hand_lms)
                        _, buffer = cv2.imencode('.jpg', clean_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        await send({'type': 'websocket.send', 'text': json.dumps({"image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}", "detections": detections, "gesture_image": gesture_image_name})})
            except: break

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    serve(port=port)
