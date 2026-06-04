#!/usr/bin/env python3
"""
Quick test script for voice recognition functionality
"""

import speech_recognition as sr
import pyperclip

def test_microphones():
    print("Testing microphone detection...")
    audio = sr.Microphone()
    print(f"Available devices: {audio.list_microphone_names()}")
    return len(audio.list_microphone_names()) > 0

def test_recognition():
    print("\nTesting speech recognition...")
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Say something (you have 5 seconds)...")
            audio = r.listen(source, timeout=5, phrase_time_limit=5)

            print("Recognizing...")
            text = r.recognize_sphinx(audio)
            print(f"Recognized: {text}")

            # Test clipboard
            pyperclip.copy(text)
            print("Copied to clipboard!")

            return True
    except sr.UnknownValueError:
        print("Could not understand audio")
        return False
    except sr.RequestError as e:
        print(f"Recognition error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Voice Recognition Test")
    print("======================")

    if test_microphones():
        print("✅ Microphones detected")
    else:
        print("❌ No microphones found")

    if test_recognition():
        print("✅ Speech recognition working")
    else:
        print("❌ Speech recognition failed")

    print("\nTest complete!")