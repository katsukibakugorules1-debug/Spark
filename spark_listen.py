"""Microphone-based, experimental speaker-verified listener for SPARK.

Install dependencies with:
    python -m pip install -r requirements.txt

Speaker matching runs locally. After a match, speech-to-text uses Google's
online recognition service and sends the captured audio to Google. The script
asks before starting. Voice matching is experimental, not secure authentication.
"""

import tempfile


def listen() -> None:
    """Recognize spoken commands and forward them to SPARK Core."""
    try:
        import speech_recognition as sr
    except ImportError:
        print(
            "Voice input dependencies are missing. Install them with:\n"
            "python -m pip install SpeechRecognition PyAudio"
        )
        return

    from spark_core import interpret
    try:
        from spark_voiceprint import (
            DEFAULT_VOICEPRINT_PATH,
            is_authorized_speaker,
        )
    except ImportError as error:
        print(f"Speaker verification dependencies are missing: {error}")
        print("Install them with: python -m pip install -r requirements.txt")
        return

    if not DEFAULT_VOICEPRINT_PATH.is_file():
        print(
            f"Master voiceprint not found at {DEFAULT_VOICEPRINT_PATH}. "
            "Create it first with: python spark_voiceprint.py"
        )
        return

    print(
        "Voice matching runs locally. After a match, speech transcription uses "
        "Google's online recognition service and sends the captured audio to Google."
    )
    if input("Continue? Type 'yes' to allow voice transcription: ").strip().lower() != "yes":
        print("SPARK voice listener cancelled.")
        return

    print("Experimental local ECAPA voice check enabled.")
    recognizer = sr.Recognizer()
    print("Requesting microphone access...")

    try:
        with sr.Microphone() as microphone:
            print("Calibrating microphone. Please stay quiet for a moment.")
            recognizer.adjust_for_ambient_noise(microphone, duration=1)
            print("SPARK voice listener ready. Speak a command; say 'exit' to stop.")

            while True:
                try:
                    audio = recognizer.listen(
                        microphone,
                        timeout=5,
                        phrase_time_limit=8,
                    )
                except sr.WaitTimeoutError:
                    continue

                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav") as sample_file:
                        sample_file.write(audio.get_wav_data())
                        sample_file.flush()
                        verified = is_authorized_speaker(sample_file.name)
                except Exception as error:
                    print(f"Local voice check failed: {error}")
                    continue

                if not verified:
                    print("SPARK: Unauthorized voice detected.")
                    continue

                print("SPARK: Voice verified. Hello!")

                try:
                    command = recognizer.recognize_google(audio).strip()
                except sr.UnknownValueError:
                    print("I didn't catch that. Please try again.")
                    continue
                except sr.RequestError as error:
                    print(f"Speech recognition service error: {error}")
                    return

                if not command:
                    continue

                print(f"You said: {command}")
                if command.lower() in {"exit", "quit", "stop listening"}:
                    print("SPARK voice listener stopped.")
                    return

                print(f"SPARK: {interpret(command)}")
    except (OSError, AttributeError) as error:
        print(f"Could not access the microphone: {error}")
        print("Check that a microphone is connected and permitted for this terminal.")

if __name__ == "__main__":
    listen()
