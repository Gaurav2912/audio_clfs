import torch
import whisper
import warnings
import torchaudio
import numpy as np
from PIL import Image
import soundfile as sf
import tensorflow as tf
import csv, pyaudio, resampy
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn.functional import softmax
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Suppress specific UserWarning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


def whisper_help_mic(wave_file, result_queue, max_duration=5):
    model_wh = whisper.load_model("tiny.en")
    audio = whisper.load_audio(wave_file)
    chunk = audio
    
    # Pad the chunk if it's shorter than 2 seconds
    if len(chunk) < 32000:  # 2 seconds at 16kHz
        chunk = np.pad(chunk, (0, 32000 - len(chunk)))
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model_wh.transcribe, chunk, language="en")
        try:
            result = future.result(timeout=max_duration)
            transcription = result['text'].lower()
            print(transcription)
            key_words = ['hel', 'hey', 'hope','head']
            detected = any(word in transcription for word in key_words)
            
            if detected:
                result_queue.put(("HELP AUDIO DETECTED", 1.0))
            else:
                result_queue.put(("BACKGROUND NOISE", 0.0))
        
        except TimeoutError:
            result_queue.put(("BACKGROUND NOISE", 0.0))
            print("TimeoutError")

# Audio classification functions
def function_shout_mic(wave_file, result_queue):
    model_resnet = "./models/Resnet34_Model_2024-07-25--10-36-34.pt"

    def pad_waveform(waveform, target_length):
        num_channels, current_length = waveform.shape
        if current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def transform_data_to_image(audio, sample_rate):
        audio = torch.cat([audio] * 5, dim=-1)
        audio = pad_waveform(audio, 160000)
        spectrogram_tensor = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64, n_fft=1024)(audio)[0] + 1e-10
        image_path = './audio_img.png'
        plt.imsave(image_path, spectrogram_tensor.log2().numpy(), cmap='viridis')
        return image_path

    transform = transforms.Compose([
        transforms.Resize((64, 313)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :])
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    model = torch.load(model_resnet, map_location=device)
    model = model.to(device)
    model.eval()

    pred_dict = {0: "Background Noise", 1: "Shout"}
    thresh = 0.95

    audio, sample_rate = torchaudio.load(wave_file)
    image_path = transform_data_to_image(audio, sample_rate)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image.to(device))
    prob = max(softmax(outputs, dim=1).cpu().detach().numpy().ravel())
    predict = outputs.argmax(dim=1).cpu().detach().numpy().ravel()[0]

    if predict == 1 and prob >= thresh:
        result = "DISTRESSED AUDIO DETECTED"
    else:
        result = "BACKGROUND NOISE"

    result_queue.put((result, float(prob)))

def run_yamnet_mic(filepath, result_queue):
    interpreter = tf.lite.Interpreter("./models/yamnet_16000.tflite")
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    threshold = 0.5
    CHUNK = 16000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    with open("./models/yamnet_class_map.csv") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        yamnet_classes = np.array([display_name for (_, _, display_name) in reader])

    wav_data, sr = sf.read(filepath, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != 16000:
        waveform = resampy.resample(waveform, sr, 16000)
    waveform = waveform[:16000]

    interpreter.set_tensor(inputs[0]['index'], np.expand_dims(np.array(waveform, dtype=np.float32), axis=0))
    interpreter.invoke()
    scores = interpreter.get_tensor(outputs[0]['index'])
    prediction = np.mean(scores, axis=0)
    max_index = np.argmax(prediction)

    if max_index in [420, 421, 422, 423, 424, 425]:
        out = "GUN SHOT DETECTED"
    else:
        out = "BACKGROUND NOISE"
    print(out)
    print(max_index)

    result_queue.put((out, float(prediction[max_index])))

