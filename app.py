from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import threading
import time
from queue import Queue
import sounddevice as sd
import wavio as wv
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from werkzeug.utils import secure_filename
import soundfile as sf
import pygame
import whisper

# Import your classification functions here
from wav_file_script import function_shout_wav, run_yamnet_wav, whisper_help_wav
from microphone_script import function_shout_mic, run_yamnet_mic, whisper_help_mic


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables
prediction_running = False
prediction_thread = None
wav_file = None
latest_result = "No prediction yet"

# Initialize pygame mixer for file playback
pygame.mixer.init()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/microphone')
def microphone():
    return render_template('microphone.html')

@app.route('/file')
def file():
    return render_template('file.html')

@app.route('/start_microphone_prediction', methods=['POST'])
def start_microphone_prediction():
    global prediction_running, prediction_thread
    prediction_running = True
    prediction_thread = threading.Thread(target=predict_microphone)
    prediction_thread.start()
    return jsonify({"status": "started"})

@app.route('/start_file_prediction', methods=['POST'])
def start_file_prediction():
    global prediction_running, prediction_thread, wav_file
    if not wav_file:
        return jsonify({"error": "Please select a WAV file first."})
    prediction_running = True
    pygame.mixer.music.load(wav_file)
    pygame.mixer.music.play()
    prediction_thread = threading.Thread(target=predict_file)
    prediction_thread.start()
    return jsonify({"status": "started"})

@app.route('/stop_prediction', methods=['POST'])
def stop_prediction():
    global prediction_running, prediction_thread
    prediction_running = False
    if wav_file:
        pygame.mixer.music.stop()
    if prediction_thread:
        prediction_thread.join()
        prediction_thread = None
    return jsonify({"status": "stopped"})

@app.route('/upload_file', methods=['POST'])
def upload_file():
    global wav_file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        wav_file = file_path
        return jsonify({"filename": filename})
    return jsonify({"error": "Invalid file type"})

@app.route('/get_latest_result')
def get_latest_result():
    global latest_result
    return jsonify({"result": latest_result})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_microphone():
    global prediction_running, latest_result
    freq = 16000
    duration = 2

    while prediction_running:
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
        sd.wait()
        file_path_16 = "temp_recording_16.wav"
        wv.write(file_path_16, recording, freq, sampwidth=2)

        result_queue = Queue()

        shout_thread = threading.Thread(target=function_shout_mic, args=(file_path_16, result_queue))
        gunshot_thread = threading.Thread(target=run_yamnet_mic, args=(file_path_16, result_queue))
        help_thread = threading.Thread(target=whisper_help_mic, args=(file_path_16, result_queue))

        shout_thread.start()
        gunshot_thread.start()
        help_thread.start()

        shout_thread.join()
        gunshot_thread.join()
        help_thread.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        latest_result = update_detection_results(results)
        time.sleep(0.5)

def predict_file():
    global prediction_running, wav_file, latest_result
    wav_data, sr = sf.read(wav_file, dtype=np.int16)
    duration = len(wav_data) // sr
    chunk_duration = 2  # 2 seconds
    chunk_samples = chunk_duration * sr
    is_sleep = True
    model_wh = whisper.load_model("tiny.en")
    for chunk_start in range(0, len(wav_data), chunk_samples):
        if not prediction_running:
            break
        chunk_end = chunk_start + chunk_samples
        if chunk_end > len(wav_data):
            chunk_end = len(wav_data)

        result_queue = Queue()
        
        tic = time.perf_counter()
        shout_thread = threading.Thread(target=function_shout_wav, args=(wav_file, result_queue, chunk_start, chunk_end))
        gunshot_thread = threading.Thread(target=run_yamnet_wav, args=(wav_file, result_queue, chunk_start, chunk_end))
        help_thread = threading.Thread(target=whisper_help_wav, args=(wav_file, result_queue, chunk_start, chunk_end , model_wh))
        
        shout_thread.start()
        gunshot_thread.start()
        help_thread.start()

        shout_thread.join()
        gunshot_thread.join()
        help_thread.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        print(results)
        # for result, prob in results:
        #     print(f"{result}: {prob}")
        tac = time.perf_counter()
        print(f"{tac-tic}, time taken.")

        latest_result = update_detection_results(results)

        if ((tac-tic)<=2) and is_sleep:
            time.sleep(max(1.85 - (tac-tic), 0))
        is_sleep = True
        if ((tac-tic)>2):
            is_sleep = False

    prediction_running = False
    latest_result = "Audio processing completed"

def update_detection_results(results):
    gunshot_result = "BACKGROUND NOISE"
    shout_result = "BACKGROUND NOISE"
    help_result = "BACKGROUND NOISE"
    gunshot_prob = 0.0
    shout_prob = 0.0
    help_prob = 0.0

    for result, prob in results:
        if result == "GUN SHOT DETECTED":
            gunshot_result = result
            gunshot_prob = prob
        elif result == "DISTRESSED AUDIO DETECTED":
            shout_result = result
            shout_prob = prob
        elif result == "HELP AUDIO DETECTED":
            help_result = result
            help_prob = prob

        if gunshot_result == "BACKGROUND NOISE" and shout_result == "BACKGROUND NOISE" and help_result == "BACKGROUND NOISE":
            detection_result = "BACKGROUND NOISE"
        else:
            # At least one positive detection
            positive_results = []
            if gunshot_result != "BACKGROUND NOISE":
                positive_results.append((gunshot_result, gunshot_prob))
            if shout_result != "BACKGROUND NOISE":
                positive_results.append((shout_result, shout_prob))
            if help_result != "BACKGROUND NOISE":
                positive_results.append((help_result, help_prob))

            # Sort by probability in descending order
            positive_results.sort(key=lambda x: x[1], reverse=True)
            
            # The result with the highest probability is the first in the sorted list
            detection_result = positive_results[0][0]
    
    return detection_result

def down_sample(input_file):
    target_sample_rate = 32000
    original_sample_rate, audio_data = wav.read(input_file)
    number_of_samples = round(len(audio_data) * float(target_sample_rate) / original_sample_rate)
    resampled_audio = resample(audio_data, number_of_samples)
    output_file = 'temp_recording_32.wav'
    wav.write(output_file, target_sample_rate, resampled_audio.astype(audio_data.dtype))
    return output_file

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=8001, host='0.0.0.0')