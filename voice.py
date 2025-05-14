import speech_recognition as sr

# The names we'll cycle through
REPLIES = ["Vaibhav", "Manas", "Dev"]

def listen_for_who_is_he(recognizer, mic):
    """
    Listen on the mic, return the transcribed text in lowercase.
    Returns empty string on failure.
    """
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("🎤 Listening… (say 'who is he')")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        # Couldn’t catch your words
        return ""
    except sr.RequestError as e:
        print(f"⚠️ Recognizer error: {e}")
        return ""

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    count = 0
    print("🌟 Ready. Speak “who is he” to unveil a name. Ctrl+C to exit.\n")

    try:
        while True:
            text = listen_for_who_is_he(recognizer, mic)
            if "who is he" in text:
                name = REPLIES[count % len(REPLIES)]
                print(f"🤖 → {name}\n")
                count += 1
            # otherwise, remain still, listening for your next invocation
    except KeyboardInterrupt:
        print("\n✨ Farewell.")

if __name__ == "__main__":
    main()
