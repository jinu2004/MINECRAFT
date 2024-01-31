import pyaudio
from queue import Queue
import subprocess
import json
from vosk import Model, KaldiRecognizer
import time
from threading import Thread



# p = pyaudio.PyAudio()
# for i in range(p.get_device_count()):
#     print(p.get_device_info_by_index(i))

# p.terminate()


messages = Queue()
recordings = Queue()
CHANNELS = 2
FRAME_RATE = 48000
RECORD_SECONDS = 20
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_SIZE = 2

model = Model(model_name="vosk-model-en-us-0.22")
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)



def record_microphone(chunk=1024):
    p = pyaudio.PyAudio()

    stream = p.open(format=AUDIO_FORMAT,
                    channels=CHANNELS,
                    rate=FRAME_RATE,
                    input=True,
                    input_device_index=6,
                    frames_per_buffer=chunk)

    frames = []

    while not messages.empty():
        data = stream.read(chunk)
        frames.append(data)
        if len(frames) >= (FRAME_RATE * RECORD_SECONDS) / chunk:
            recordings.put(frames.copy())
            frames = []

    stream.stop_stream()
    stream.close()
    p.terminate()
    
def speech_recognition():
    
    while not messages.empty():
        frames = recordings.get()
        
        rec.AcceptWaveform(b''.join(frames))
        result = rec.Result()
        text = json.loads(result)["text"]
        
        cased = subprocess.check_output('python recasepunc/recasepunc.py predict recasepunc/checkpoint', shell=True, text=True, input=text)
        # output.append_stdout(cased)
        print(cased)
        time.sleep(1)


def start_recording():
    messages.put(True)
    print("Starting...")
    record = Thread(target=record_microphone)
    record.start()
    transcribe = Thread(target=speech_recognition)
    transcribe.start()

def stop_recording():
    with True:
        messages.get()
        print("Stopped.")
        
        
start_recording()  
     