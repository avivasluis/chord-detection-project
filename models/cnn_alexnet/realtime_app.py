import streamlit as st
import os
import sys
import json
import numpy as np
import librosa
import onnxruntime as ort
from PIL import Image
import sounddevice as sd
import queue
import time

# Add project root to path for n-gram model import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.n_gram_chord_progression.ngram_model import NGramModel

# Set page configuration
st.set_page_config(
    page_title="Real-time Chord Detection",
    page_icon="🎸",
    layout="wide"
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
NGRAM_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'n_gram_chord_progression', 'ngram_model.pkl')

@st.cache_resource
def load_ngram_model():
    """Load the n-gram model for chord progression prediction."""
    if os.path.exists(NGRAM_MODEL_PATH):
        return NGramModel.load(NGRAM_MODEL_PATH)
    return None

def convert_chord_for_ngram(chord_name):
    """
    Convert CNN chord names to n-gram model format.
    CNN format examples: 'Am', 'C', 'Dm', 'E', 'G', etc.
    N-gram format: 'C', 'Am', 'G7', etc. (root or root+m for minor)
    """
    # Most CNN chord names should already be compatible
    # Just ensure consistency (strip whitespace, etc.)
    return chord_name.strip()

def get_available_versions():
    """Scan results directory for available model versions."""
    if not os.path.exists(RESULTS_DIR):
        return []
    versions = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    # Filter for versions that have the required files
    valid_versions = []
    for v in versions:
        model_path = os.path.join(RESULTS_DIR, v, 'alexnet_chord_classifier.onnx')
        config_path = os.path.join(RESULTS_DIR, v, 'chord_model_config.json')
        if os.path.exists(model_path) and os.path.exists(config_path):
            valid_versions.append(v)
    return sorted(valid_versions)

def load_model(version):
    """Load the model and config for a specific version."""
    model_dir = os.path.join(RESULTS_DIR, version)
    config_path = os.path.join(model_dir, 'chord_model_config.json')
    model_path = os.path.join(model_dir, 'alexnet_chord_classifier.onnx')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    session = ort.InferenceSession(model_path)
    return session, config

def preprocess_audio_chunk(audio_chunk, sr, config):
    """
    Preprocess an audio chunk for chord classification.
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
    except Exception as e:
        # Fallback for silence or too short audio
        # Create a dummy chromagram of correct shape if CQT fails
        # This can happen if the buffer is silent
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

def main():
    st.title("🎸 Real-time Chord Detection")
    st.markdown("Connect your guitar or use your microphone to test the models.")

    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # Version Selection
    versions = get_available_versions()
    if not versions:
        st.error("No model versions found in 'results/' directory.")
        return
        
    selected_version = st.sidebar.selectbox("Select Model Version", versions, index=len(versions)-1)
    
    # Load Model
    try:
        session, config = load_model(selected_version)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        st.sidebar.success(f"Loaded {selected_version}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Load N-gram model for chord progression prediction
    ngram_model = load_ngram_model()
    if ngram_model:
        st.sidebar.success("N-gram model loaded")
    else:
        st.sidebar.warning("N-gram model not found - recommendations disabled")

    # Audio Device Selection
    devices = sd.query_devices()
    
    # Build list of input devices (microphones, virtual cables, etc.)
    input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    
    # Add info about capturing system audio
    st.sidebar.markdown("### Audio Source")
    st.sidebar.info(
        "💡 **To capture system audio** (VLC, Chrome, etc.):\n"
        "- Enable 'Stereo Mix' in Windows Sound settings, OR\n"
        "- Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) and set it as default output"
    )
    
    # Try to find a default input device
    default_input = sd.default.device[0]
    default_index = 0
    for i, dev_str in enumerate(input_devices):
        if str(default_input) in dev_str.split(':')[0]:
            default_index = i
            break
    
    # Check for common loopback/virtual cable devices and highlight them
    loopback_keywords = ['stereo mix', 'cable output', 'virtual', 'loopback', 'what u hear', 'voicemeeter']
    for i, dev_str in enumerate(input_devices):
        if any(kw in dev_str.lower() for kw in loopback_keywords):
            default_index = i  # Auto-select loopback device if found
            break
            
    selected_device_str = st.sidebar.selectbox("Input Device", input_devices, index=default_index)
    
    # Show hint if a loopback device is selected
    if any(kw in selected_device_str.lower() for kw in loopback_keywords):
        st.sidebar.success("✅ Loopback device detected - system audio capture enabled!")
    device_id = int(selected_device_str.split(':')[0])

    # Audio Parameters
    sample_rate = config.get('sample_rate', 44100) # Use model's sample rate
    duration = config.get('duration', 1.0) # Window duration needed for model
    buffer_size = int(sample_rate * duration)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Sample Rate: {sample_rate} Hz\nWindow Duration: {duration}s")
    
    # Prediction Stability Settings
    st.sidebar.markdown("### Prediction Stability")
    smoothing_factor = st.sidebar.slider(
        "Smoothing Factor (EMA)", 
        min_value=0.0, max_value=0.95, value=0.7, step=0.05,
        help="Higher = more stable but slower to react. 0 = no smoothing."
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, max_value=0.9, value=0.3, step=0.05,
        help="Minimum confidence to display a prediction."
    )
    change_threshold = st.sidebar.slider(
        "Change Threshold",
        min_value=0.0, max_value=0.5, value=0.1, step=0.05,
        help="New chord must exceed current by this margin to switch."
    )

    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        start_button = st.button("Start Listening", type="primary")
        stop_button = st.button("Stop")
        
    placeholder = st.empty()
    
    if 'listening' not in st.session_state:
        st.session_state.listening = False
    if 'smoothed_probs' not in st.session_state:
        st.session_state.smoothed_probs = None
    if 'current_chord' not in st.session_state:
        st.session_state.current_chord = None
    if 'chord_history' not in st.session_state:
        st.session_state.chord_history = []  # List of previous chords
        
    if start_button:
        st.session_state.listening = True
        # Reset smoothing state when starting fresh
        st.session_state.smoothed_probs = None
        st.session_state.current_chord = None
        st.session_state.chord_history = []
        
    if stop_button:
        st.session_state.listening = False

    if st.session_state.listening:
        # Audio processing loop
        audio_queue = queue.Queue()
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            audio_queue.put(indata.copy())

        # Start stream
        with sd.InputStream(device=device_id, channels=1, samplerate=sample_rate, 
                           blocksize=int(sample_rate * 0.5), # Read in smaller chunks
                           callback=audio_callback):
            
            # Buffer to hold enough audio for the model window
            audio_buffer = np.zeros(buffer_size, dtype=np.float32)
            
            while st.session_state.listening:
                try:
                    # Get new data
                    if not audio_queue.empty():
                        new_data = audio_queue.get()
                        new_data = new_data.flatten()
                        
                        # Shift buffer and append new data
                        audio_buffer = np.roll(audio_buffer, -len(new_data))
                        audio_buffer[-len(new_data):] = new_data
                        
                        # Process only if we have filled the buffer (or mostly filled)
                        # We just run continuously on the sliding window
                        
                        # Run Inference
                        input_tensor = preprocess_audio_chunk(audio_buffer, sample_rate, config)
                        
                        if input_tensor is not None:
                            outputs = session.run([output_name], {input_name: input_tensor})
                            logits = outputs[0][0]
                            
                            # Softmax
                            exp_logits = np.exp(logits - np.max(logits))
                            raw_probs = exp_logits / exp_logits.sum()
                            
                            # Apply Exponential Moving Average (EMA) smoothing
                            if st.session_state.smoothed_probs is None:
                                st.session_state.smoothed_probs = raw_probs.copy()
                            else:
                                # EMA: smoothed = alpha * smoothed + (1 - alpha) * new
                                st.session_state.smoothed_probs = (
                                    smoothing_factor * st.session_state.smoothed_probs + 
                                    (1 - smoothing_factor) * raw_probs
                                )
                            
                            probs = st.session_state.smoothed_probs
                            pred_idx = np.argmax(probs)
                            confidence = probs[pred_idx]
                            new_chord_name = config['idx_to_class'][str(pred_idx)]
                            
                            # Apply hysteresis: only change chord if significantly better
                            display_chord = st.session_state.current_chord
                            
                            if display_chord is None:
                                display_chord = new_chord_name
                                st.session_state.current_chord = new_chord_name
                            elif new_chord_name != display_chord:
                                # Find the index of the current chord
                                current_idx = None
                                for idx, name in config['idx_to_class'].items():
                                    if name == display_chord:
                                        current_idx = int(idx)
                                        break
                                
                                # Only switch if new chord's probability exceeds current by threshold
                                if current_idx is not None:
                                    current_prob = probs[current_idx]
                                    if confidence > current_prob + change_threshold:
                                        # Add old chord to history before switching
                                        st.session_state.chord_history.append(display_chord)
                                        # Keep only last 2 chords in history
                                        if len(st.session_state.chord_history) > 2:
                                            st.session_state.chord_history = st.session_state.chord_history[-2:]
                                        display_chord = new_chord_name
                                        st.session_state.current_chord = new_chord_name
                                else:
                                    st.session_state.chord_history.append(display_chord)
                                    if len(st.session_state.chord_history) > 2:
                                        st.session_state.chord_history = st.session_state.chord_history[-2:]
                                    display_chord = new_chord_name
                                    st.session_state.current_chord = new_chord_name
                            
                            # Get top 3 from smoothed probs
                            top_indices = np.argsort(probs)[::-1][:3]
                            top_predictions = [(config['idx_to_class'][str(i)], probs[i]) for i in top_indices]
                            
                            # Get N-gram recommendation based on previous chord + current chord
                            ngram_recommendations = []
                            if ngram_model and display_chord:
                                # Build sequence for n-gram: [previous_chord, current_chord]
                                # Trigram model needs exactly 2 chords to predict the 3rd
                                history = st.session_state.chord_history
                                ngram_sequence = []
                                
                                if len(history) >= 1:
                                    # Use the most recent chord from history + current
                                    ngram_sequence = [
                                        convert_chord_for_ngram(history[-1]),
                                        convert_chord_for_ngram(display_chord)
                                    ]
                                else:
                                    # Only current chord available
                                    ngram_sequence = [convert_chord_for_ngram(display_chord)]
                                
                                # Get top 3 recommendations
                                try:
                                    ngram_recommendations = ngram_model.predict_next(
                                        ngram_sequence, 
                                        top_k=3,
                                        use_smoothing=True
                                    )
                                except Exception as e:
                                    pass  # Silently fail if prediction fails
                            
                            # Update UI
                            with placeholder.container():
                                # Build history display string
                                history = st.session_state.chord_history
                                if len(history) >= 2:
                                    history_display = f"{history[-2]} → {history[-1]}"
                                elif len(history) == 1:
                                    history_display = f"— → {history[-1]}"
                                else:
                                    history_display = "— → —"
                                
                                if confidence >= confidence_threshold:
                                    st.markdown(
                                        f"""
                                        <div style="text-align: center; padding: 20px;">
                                            <p style="font-size: 24px; color: #888; margin-bottom: 10px;">
                                                {history_display} →
                                            </p>
                                            <h1 style="font-size: 100px; margin-bottom: 0;">{display_chord}</h1>
                                            <h3 style="color: gray; margin-top: 0;">Confidence: {confidence*100:.1f}%</h3>
                                        </div>
                                        """, unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f"""
                                        <div style="text-align: center; padding: 20px;">
                                            <p style="font-size: 24px; color: #888; margin-bottom: 10px;">
                                                {history_display} →
                                            </p>
                                            <h1 style="font-size: 100px; margin-bottom: 0; color: #666;">?</h1>
                                            <h3 style="color: gray; margin-top: 0;">Low confidence ({confidence*100:.1f}%)</h3>
                                        </div>
                                        """, unsafe_allow_html=True
                                    )
                                
                                # Show N-gram recommended next chord
                                if ngram_recommendations:
                                    st.markdown("---")
                                    st.markdown("### 🎵 Suggested Next Chord")
                                    
                                    # Display top recommendation prominently
                                    top_rec = ngram_recommendations[0]
                                    st.markdown(
                                        f"""
                                        <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 10px; margin-bottom: 15px;">
                                            <h2 style="font-size: 48px; margin: 0; color: #4ecca3;">{top_rec[0]}</h2>
                                            <p style="color: #888; margin: 5px 0 0 0;">Probability: {top_rec[1]*100:.1f}%</p>
                                        </div>
                                        """, unsafe_allow_html=True
                                    )
                                    
                                    # Show other suggestions
                                    if len(ngram_recommendations) > 1:
                                        st.write("Other suggestions:")
                                        cols = st.columns(len(ngram_recommendations) - 1)
                                        for i, (chord, prob) in enumerate(ngram_recommendations[1:]):
                                            with cols[i]:
                                                st.metric(label=chord, value=f"{prob*100:.1f}%")
                                
                                st.markdown("---")
                                
                                # Show top 3 bars
                                st.write("Top Predictions (smoothed):")
                                for chord, prob in top_predictions:
                                    st.progress(float(prob), text=f"{chord} ({prob*100:.1f}%)")
                        else:
                            with placeholder.container():
                                st.warning("Audio signal too weak or invalid...")
                                
                    else:
                        time.sleep(0.01) # Sleep briefly if no new audio
                        
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    break
                

if __name__ == "__main__":
    main()

