from src.utils import get_model, RealTimeVisionSystem
from src.downloader import download_video
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import torch
import shutil
import json
from pyngrok import ngrok, conf
import threading
import time


def start_ngrok():
    conf.get_default().config_path = "ngrok.yaml"
    
    tunnel = ngrok.connect(name="myapp")
    print(f"Ngrok tunnel started at: {tunnel.public_url}")
    
    # Giữ đường hầm mở
    while True:
        time.sleep(1)

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model
model, face_detector, face_predictor = get_model()
threshold_fake = (0.31, 0.66)
# threshold_fake = (0.35, 0.75)


@app.route('/')
def home():
    try:
        if not os.path.exists('templates/index.html'):
            raise Exception("Template file 'index.html' not found")
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"{e}")
        return "Error", 500

@app.route('/download', methods=['POST'])
@cross_origin()
def download():
    try:
        data = request.json
        if 'url' not in data:
            return jsonify({'error': 'Missing URL'}), 400

        url = data['url']
        file_path, _ = download_video(url)
        return jsonify({'file_path': file_path})
    except Exception as e:
        app.logger.error(f"Error downloading video: {e}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/process', methods=['POST'])
@cross_origin()
def process_video():
    start_time = time.time()
    shutil.rmtree("static")
    os.makedirs("static", exist_ok=True)
    try:
        data = request.json
        if 'url' not in data:
            return jsonify({'error': 'Missing URL'}), 400

        url = data['url']
        file_path, _ = download_video(url)  
        process_video = RealTimeVisionSystem(model, face_detector, face_predictor, file_path, threshold_fake, None, 30, "fixed_stride")
        video_file, prob_file, imgs_folder, result = process_video.run()
        os.remove(file_path)

        # Trả về URL cho video output
        video_name = os.path.basename(video_file)
        end_time = time.time()
        time_process = end_time - start_time
        return jsonify({'output_path': f"https://firefly-genuine-adequately.ngrok-free.app/output?v={video_name.split('.')[0]}&time_process={time_process:.2f}",
                        'result': f"{result}",
                        'time_process': f"{time_process}"})

    except Exception as e:
        app.logger.error(f"Error processing video: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/output', methods=['GET'])
def render_video():
    try:
        param = request.args.get('v')
        time_process = request.args.get('time_process')
        print(f"time_process: {time_process}")
        video_path = f"videos/{param}.mp4"
        imgs_path = [f"images/{param}/{img_path}" for img_path in os.listdir(f"static/images/{param}")]

        prop_path = f"static/props/{param}.json"
        with open(prop_path, 'r') as f:
            props_item = json.load(f)
        
        probs_data = []
        for img_path in imgs_path:
            id_ = os.path.basename(img_path).split('.')[0]
            probs_data.append(props_item[id_])

        return render_template('output.html', video_path=video_path, props_data=probs_data, imgs_path=imgs_path, threshold_fake=threshold_fake, time_processing=time_process)
    except Exception as e:
        app.logger.error(f"Error rendering video: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    # Chạy ngrok trong một thread riêng
    ngrok_thread = threading.Thread(target=start_ngrok)
    ngrok_thread.daemon = True
    ngrok_thread.start()


    # context = ('HTTPS/cert.pem', 'HTTPS/key.pem')
    # app.run(host='0.0.0.0', debug=True, port=5000, ssl_context=context)
    app.run(host='0.0.0.0', debug=True, port=5000)
