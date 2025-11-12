from kokoro import KPipeline
import numpy as np
import soundfile as sf

KOKORO_VOICES = ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis']

def load_numpy_kpipeline():
    import numpy as np
    return np, KPipeline



class KokoroTTS:
    def __init__(self, lang_code="a", speed=1.0, output_format="WAV", device="cpu"):
        # lang code: {'a': 'American English', 'b': 'British English', 'e': 'es', 'f': 'fr-fr', 'h': 'hi', 'i': 'it', 'p': 'pt-br', 'j': 'Japanese', 'z': 'Mandarin Chinese'}
        self.lang_code = lang_code
        self.speed = speed
        self.output_format = output_format
        self.device = device
        self.KPipeline = KPipeline
        self.tts = self.KPipeline(
            lang_code=self.lang_code, repo_id="hexgrad/Kokoro-82M", device=self.device
        )
        
    def synthesize(self, text, voice, output_path, split_pattern=None):
        """
        Generate speech from text and save it as a WAV file.

        :param text: The text to be synthesized.
        :param voice: The voice model to use for synthesis.
        :param output_path: Path to save the WAV file.
        :param split_pattern: Optional pattern to split text into smaller chunks.
        """

        generator = self.tts(
            text, voice=voice, speed=self.speed, split_pattern=split_pattern
        )
        audio_segments = []
        for gs, ps, audio in generator:
            if audio is not None:
                    # Only convert if it's a numpy array, not if already tensor
                    audio_tensor = audio 

                    audio_segments.append(audio_tensor)
        audio = np.concatenate(audio_segments)
        
        sf.write(output_path, audio, 24000, format=self.output_format)