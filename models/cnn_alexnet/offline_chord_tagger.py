#!/usr/bin/env python3
"""
Offline Chord Tagger

Analyzes audio files and produces chord annotations with timestamps,
using bidirectional median filtering for robust predictions.

Usage:
    python offline_chord_tagger.py input_audio.wav -o output.txt --hop 0.25 --median-kernel 5
"""

import argparse
import os
import sys
import json
import numpy as np
import librosa
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import median_filter
from typing import List, Tuple, Optional

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DEFAULT_VERSION = 'v10'


def load_model(version: str) -> Tuple[ort.InferenceSession, dict]:
    """Load the ONNX model and config for a specific version."""
    model_dir = os.path.join(RESULTS_DIR, version)
    config_path = os.path.join(model_dir, 'chord_model_config.json')
    model_path = os.path.join(model_dir, 'alexnet_chord_classifier.onnx')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    session = ort.InferenceSession(model_path)
    return session, config


def preprocess_audio_chunk(audio_chunk: np.ndarray, sr: int, config: dict) -> Optional[np.ndarray]:
    """
    Preprocess an audio chunk for chord classification.
    Replicates the exact preprocessing from training.
    """
    hop_length = config.get('hop_length', 512)
    input_size = tuple(config['input_size'])
    normalize_mean = config['normalize_mean']
    normalize_std = config['normalize_std']
    
    # Ensure audio is float32
    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32)

    # Compute CQT chromagram
    try:
        chromagram = librosa.feature.chroma_cqt(
            y=audio_chunk, 
            sr=sr, 
            hop_length=hop_length,
            n_chroma=12,
            n_octaves=7,
        )
    except Exception:
        # Fallback for silence or too short audio
        return None
    
    # Convert to dB scale
    chromagram_db = librosa.amplitude_to_db(chromagram + 1e-10, ref=np.max)
    
    # Normalize to 0-255 range
    norm_feature = (chromagram_db - chromagram_db.min()) / (chromagram_db.max() - chromagram_db.min() + 1e-10)
    norm_feature = (norm_feature * 255).astype(np.uint8)
    
    # Convert to PIL Image and resize
    img = Image.fromarray(norm_feature)
    img = img.convert("RGB")
    img = img.resize(input_size, Image.Resampling.BILINEAR)
    
    # Convert to numpy and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Apply normalization
    for i in range(3):
        img_array[:, :, i] = (img_array[:, :, i] - normalize_mean[i]) / normalize_std[i]
    
    # Transpose to (C, H, W) and add batch dimension
    img_tensor = np.transpose(img_array, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    return img_tensor.astype(np.float32)


def extract_windows(audio: np.ndarray, sr: int, window_duration: float, hop_duration: float) -> List[Tuple[np.ndarray, float]]:
    """
    Extract overlapping windows from audio.
    
    Returns:
        List of (audio_chunk, start_time) tuples
    """
    window_samples = int(sr * window_duration)
    hop_samples = int(sr * hop_duration)
    
    windows = []
    start_sample = 0
    
    while start_sample + window_samples <= len(audio):
        chunk = audio[start_sample:start_sample + window_samples]
        start_time = start_sample / sr
        windows.append((chunk, start_time))
        start_sample += hop_samples
    
    # Handle the last partial window if there's remaining audio
    if start_sample < len(audio) and len(audio) - start_sample > window_samples // 2:
        # Pad the last chunk to full window size
        chunk = audio[start_sample:]
        padding = window_samples - len(chunk)
        chunk = np.pad(chunk, (0, padding), 'constant')
        start_time = start_sample / sr
        windows.append((chunk, start_time))
    
    return windows


def run_inference(session: ort.InferenceSession, windows: List[Tuple[np.ndarray, float]], 
                  sr: int, config: dict) -> Tuple[np.ndarray, List[float]]:
    """
    Run inference on all windows and return probability matrix.
    
    Returns:
        prob_matrix: Shape [num_frames, num_classes]
        timestamps: List of start times for each frame
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    num_classes = config['num_classes']
    
    prob_matrix = []
    timestamps = []
    
    print(f"Processing {len(windows)} windows...")
    
    for i, (chunk, start_time) in enumerate(windows):
        input_tensor = preprocess_audio_chunk(chunk, sr, config)
        
        if input_tensor is not None:
            outputs = session.run([output_name], {input_name: input_tensor})
            logits = outputs[0][0]
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            prob_matrix.append(probs)
            timestamps.append(start_time)
        else:
            # For invalid chunks, use uniform distribution
            prob_matrix.append(np.ones(num_classes) / num_classes)
            timestamps.append(start_time)
        
        # Progress indicator
        if (i + 1) % 20 == 0 or i == len(windows) - 1:
            print(f"  Processed {i + 1}/{len(windows)} windows")
    
    return np.array(prob_matrix), timestamps


def apply_median_filter(prob_matrix: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply median filter along the time axis for bidirectional smoothing.
    
    Args:
        prob_matrix: Shape [num_frames, num_classes]
        kernel_size: Size of the median filter kernel (odd number recommended)
    
    Returns:
        Smoothed probability matrix
    """
    if kernel_size <= 1:
        return prob_matrix
    
    # Apply median filter along axis 0 (time axis) for each class
    smoothed = median_filter(prob_matrix, size=(kernel_size, 1), mode='nearest')
    
    return smoothed


def merge_segments(predictions: np.ndarray, confidences: np.ndarray, 
                   timestamps: List[float], hop_duration: float,
                   idx_to_class: dict) -> List[dict]:
    """
    Merge consecutive frames with the same chord prediction into segments.
    
    Returns:
        List of segment dictionaries with keys: chord, start_time, end_time, avg_confidence
    """
    if len(predictions) == 0:
        return []
    
    segments = []
    current_chord_idx = predictions[0]
    segment_start = timestamps[0]
    segment_confidences = [confidences[0]]
    
    for i in range(1, len(predictions)):
        if predictions[i] != current_chord_idx:
            # End current segment
            segment_end = timestamps[i]
            avg_conf = np.mean(segment_confidences)
            chord_name = idx_to_class[str(current_chord_idx)]
            
            segments.append({
                'chord': chord_name,
                'start_time': segment_start,
                'end_time': segment_end,
                'avg_confidence': avg_conf
            })
            
            # Start new segment
            current_chord_idx = predictions[i]
            segment_start = timestamps[i]
            segment_confidences = [confidences[i]]
        else:
            segment_confidences.append(confidences[i])
    
    # Don't forget the last segment
    # End time is the last timestamp plus hop duration (approximately)
    segment_end = timestamps[-1] + hop_duration
    avg_conf = np.mean(segment_confidences)
    chord_name = idx_to_class[str(current_chord_idx)]
    
    segments.append({
        'chord': chord_name,
        'start_time': segment_start,
        'end_time': segment_end,
        'avg_confidence': avg_conf
    })
    
    return segments


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.ss"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def generate_text_output(segments: List[dict], audio_path: str, 
                         confidence_threshold: float) -> str:
    """
    Generate human-readable text output.
    """
    filename = os.path.basename(audio_path)
    lines = []
    
    lines.append(f"Chord Timeline: {filename}")
    lines.append("=" * 60)
    lines.append("")
    
    # Calculate max chord name length for alignment
    max_chord_len = max(len(seg['chord']) for seg in segments) if segments else 4
    
    for seg in segments:
        start = format_time(seg['start_time'])
        end = format_time(seg['end_time'])
        chord = seg['chord']
        conf = seg['avg_confidence'] * 100
        
        # Mark low confidence predictions
        if seg['avg_confidence'] < confidence_threshold:
            chord_display = f"{chord:<{max_chord_len}} (?)"
        else:
            chord_display = f"{chord:<{max_chord_len}}    "
        
        duration = seg['end_time'] - seg['start_time']
        lines.append(f"{start} - {end}  |  {chord_display}  |  Confidence: {conf:5.1f}%  |  Duration: {duration:.2f}s")
    
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Total segments: {len(segments)}")
    
    # Summary statistics
    if segments:
        avg_conf = np.mean([s['avg_confidence'] for s in segments]) * 100
        lines.append(f"Average confidence: {avg_conf:.1f}%")
        
        low_conf_count = sum(1 for s in segments if s['avg_confidence'] < confidence_threshold)
        if low_conf_count > 0:
            lines.append(f"Low confidence segments (marked with ?): {low_conf_count}")
    
    return "\n".join(lines)


def analyze_audio(audio_path: str, version: str = DEFAULT_VERSION,
                  hop_duration: float = 0.25, median_kernel: int = 5,
                  confidence_threshold: float = 0.3) -> List[dict]:
    """
    Main analysis function.
    
    Args:
        audio_path: Path to the audio file
        version: Model version to use
        hop_duration: Hop size in seconds between windows
        median_kernel: Median filter kernel size (frames)
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        List of chord segments
    """
    print(f"Loading model {version}...")
    session, config = load_model(version)
    
    sample_rate = config.get('sample_rate', 44100)
    window_duration = config.get('duration', 1.0)
    
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    audio_duration = len(audio) / sr
    print(f"  Duration: {audio_duration:.2f}s, Sample rate: {sr}Hz")
    
    print(f"Extracting windows (window={window_duration}s, hop={hop_duration}s)...")
    windows = extract_windows(audio, sr, window_duration, hop_duration)
    print(f"  Total windows: {len(windows)}")
    
    print("Running inference...")
    prob_matrix, timestamps = run_inference(session, windows, sr, config)
    
    print(f"Applying median filter (kernel={median_kernel})...")
    smoothed_probs = apply_median_filter(prob_matrix, median_kernel)
    
    # Get predictions and confidences from smoothed probabilities
    predictions = np.argmax(smoothed_probs, axis=1)
    confidences = np.max(smoothed_probs, axis=1)
    
    print("Merging segments...")
    segments = merge_segments(predictions, confidences, timestamps, 
                              hop_duration, config['idx_to_class'])
    
    print(f"  Found {len(segments)} chord segments")
    
    return segments, config


def main():
    parser = argparse.ArgumentParser(
        description='Offline Chord Tagger - Analyze audio files and produce chord annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python offline_chord_tagger.py song.wav
  python offline_chord_tagger.py song.mp3 -o chords.txt --version v10
  python offline_chord_tagger.py song.wav --hop 0.5 --median-kernel 7
        """
    )
    
    parser.add_argument('input', help='Path to the input audio file')
    parser.add_argument('-o', '--output', help='Output file path (default: <input>_chords.txt)')
    parser.add_argument('--version', default=DEFAULT_VERSION, 
                        help=f'Model version to use (default: {DEFAULT_VERSION})')
    parser.add_argument('--hop', type=float, default=0.25,
                        help='Hop size in seconds between windows (default: 0.25)')
    parser.add_argument('--median-kernel', type=int, default=5,
                        help='Median filter kernel size in frames (default: 5)')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                        help='Minimum confidence threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_chords.txt"
    
    # Run analysis
    try:
        segments, config = analyze_audio(
            args.input,
            version=args.version,
            hop_duration=args.hop,
            median_kernel=args.median_kernel,
            confidence_threshold=args.confidence_threshold
        )
        
        # Generate and save output
        output_text = generate_text_output(segments, args.input, args.confidence_threshold)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        print(f"\nOutput saved to: {args.output}")
        print("\n" + output_text)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

