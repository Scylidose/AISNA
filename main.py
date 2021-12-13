import os
from dotenv import load_dotenv

from src.live_faces_recognition import live_faces_recognition

load_dotenv(dotenv_path="config/.env")

ENCODE_FACES = os.getenv('ENCODE_FACES')

def main():
    if ENCODE_FACES == "true":
        print("encoding")
        live_faces_recognition.encodeFaces()
    live_faces_recognition.liveRecognition()

if __name__ == "__main__":
    main()
