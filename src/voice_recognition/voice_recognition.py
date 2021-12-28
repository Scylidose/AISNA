import speech_recognition as sr

r = sr.Recognizer()

f = open("data/transcription.txt", "w")

mic = sr.Microphone()

if not isinstance(r, sr.Recognizer):
    raise TypeError("`recognizer` must be `Recognizer` instance")

if not isinstance(mic, sr.Microphone):
    raise TypeError("`microphone` must be `Microphone` instance")

def speechRecognition():
    print("listening")
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    transcript = r.recognize_google(audio)
    print(transcript)
    f.write(transcript)

    speechRecognition()
