import speech_recognition as sr

def speechRecognition(file_transcript = ""):
        
    r = sr.Recognizer()

    f = open("data/transcription.txt", "a+")

    file_transcript = f.read()

    mic = sr.Microphone()

    if not isinstance(r, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(mic, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    print("listening")
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)

            transcript = r.recognize_google(audio)
            f.write(transcript+"\n")
        f.close()
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    else:
        pass
    return speechRecognition(file_transcript)
