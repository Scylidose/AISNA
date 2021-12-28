from multiprocessing import Process
from dotenv import load_dotenv
import os
from src.voice_recognition import voice_recognition
from src.live_faces_recognition import live_faces_recognition

from utils import utils

load_dotenv(dotenv_path="config/.env")

DATA_AUGMENTATION = os.getenv('DATA_AUGMENTATION')

def live_capture():

    if DATA_AUGMENTATION == "True":
        utils.data_augmentation()
    
    live_faces_recognition.video_capture()

def speech_rec():
    voice_recognition.speechRecognition()


if __name__ == "__main__":
    p1 = Process(target=live_capture, args=())
    p2 = Process(target=speech_rec, args=())

    processes = list()
    processes.append(p1)
    processes.append(p2)

    for p in processes:
        p.start()
