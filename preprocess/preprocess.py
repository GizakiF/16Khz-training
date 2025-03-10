import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment


def load_audio_with_soundfile(input_file):
    y, sr = sf.read(input_file)
    return y, sr


# Define preprocessing function
def preprocess_audio(input_file, output_dir, segment_length=15, target_sr=16000):
    """
    Preprocesses an audio file by performing noise reduction, segmentation (15s), and downsampling (44.1kHz to 16kHz).

    Args:
        input_file (str): Path to the input audio file.
        output_dir (str): Directory to save the processed audio segments.
        segment_length (int): Length of each segment in seconds (default is 15s).
        target_sr (int): Target sampling rate (default is 16kHz).
    """

    # Load audio (original sample rate)
    y, sr = load_audio_with_soundfile(input_file)
    # y, sr = librosa.load(input_file, sr=None)

    # Apply noise reduction using spectral gating
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    # Resample from 44.1kHz to 16kHz if necessary
    if sr != target_sr:
        y_denoised = librosa.resample(
            y_denoised, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Calculate segment length in samples
    segment_samples = segment_length * sr
    total_samples = len(y_denoised)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Split and save segments
    for i, start in enumerate(range(0, total_samples, segment_samples)):
        end = start + segment_samples
        segment = y_denoised[start:end]

        # Save only full-length segments
        if len(segment) == segment_samples:
            output_file = os.path.join(output_dir, f"segment_{i + 1}.wav")
            sf.write(output_file, segment, sr)
            print(f"Saved: {output_file}")


# Example usage
# input_audio = "Predi-COVID_0098_20200624100830_1_m4a_W_0.wav"
input_audio = os.path.expanduser("~/Downloads/mobym4a.wav")
output_directory = "processed_audio"
preprocess_audio(input_audio, output_directory)
