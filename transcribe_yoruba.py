import os
import sys
import threading
from io import BytesIO

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from google import genai

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


def get_genai_client():
    """
    Return a single, configured Gemini (genai) client.
    """
    return genai.Client(
        vertexai=False,
        api_key=os.getenv("GOOGLE_AI_STUDIO"),
    )


gemini_client = get_genai_client()

fs = 16000
channels = 1
buffer_size = 1024
audio_data = np.array([])

recording_flag = False

def record_audio():
    global audio_data
    print("Recording... Press Enter to stop.")
    with sd.InputStream(callback=callback, channels=channels, samplerate=fs, blocksize=buffer_size):
        while not recording_flag:
            sd.sleep(100)

    print("Recording finished.")


def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    global audio_data
    audio_data = np.append(audio_data, indata.copy())


def toggle_recording():
    global audio_data, recording_flag
    print("Press Enter to start recording...")
    input()
    print("Recording started!")

    recording_flag = False

    record_thread = threading.Thread(target=record_audio)
    record_thread.start()

    input("Press Enter to stop recording...")
    print("Recording stopped.")

    recording_flag = True
    record_thread.join()


toggle_recording()

file_path = "mic_test.wav"
sf.write(file_path, audio_data, fs)

with open(file_path, "rb") as f:
    audio_bytes = BytesIO(f.read())

transcription = client.speech_to_text.convert(
    file=audio_bytes,
    model_id="scribe_v1",
    language_code="yo",
)

if transcription and hasattr(transcription, "text"):
    text = transcription.text
    print("Raw transcription:", text)

    prompt = f"""
    You are helping a robot understand speech commands in Yoruba.
    The robot controls a robotic arm and understands instructions like:
    - "Pick up the red screwdriver"
    - "Grab the black wrench"
    - "Move the pliers to the tray"

    However, the transcription of the Yoruba voice command may contain errors.
    Your job is to guess the intended English command based on this transcription.

    Yoruba transcription: "{text}"

    Please respond with only the most likely English instruction.
    Do not include any additional text, explanations, or clarifications.
    Just provide the instruction.
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],
        )

        if response:
            gemini_content = response.text.strip()
            if gemini_content:
                print("Gemini Interpretation:", gemini_content)
            else:
                print("Error: No content returned in Gemini response.")
        else:
            print("Error: No response from Gemini.")

    except Exception as e:
        print(f"Error with Gemini: {e}")

else:
    print("Error: Could not transcribe audio.")
