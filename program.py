import speech_recognition as sr

def start(command):
    print(command)

def stop(command):
    print(command)

def reset(command):
    print(command)

def main():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, 1)

    while True:
        with sr.Microphone() as source:
            print("Say a command:")

            try:
                audio = recognizer.listen(source, timeout=4)

                command = recognizer.recognize_google(audio).lower()
                
                if "start" in command:
                    start(command)
                elif "stop" in command:
                    stop(command)
                elif "reset" in command:
                    reset(command)
                elif "exit" in command:
                    print("Exiting program.")
                    break
                else:
                    print("Command not recognized. Try again with start, stop, reset, or exit.")
                    

            except sr.UnknownValueError:
                print("Could not understand audio. Try again.")
            except sr.RequestError as e:
                print(f"Error connecting to Google Speech Recognition service: {e}")

if __name__ == "__main__":
    main()
