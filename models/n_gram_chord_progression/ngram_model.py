import pickle
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Any, Tuple
import jams

class NGramModel:
    def __init__(self, n=3):
        self.N = n
        # We need counts for ALL orders (N, N-1, ... 1) for JM smoothing
        # structure: self.counts[order][history_tuple][chord] = count
        self.counts = {i: defaultdict(Counter) for i in range(1, n + 1)}
        self.total_counts = {i: defaultdict(int) for i in range(1, n + 1)}
        self.vocab = set()

    def train(self, sequences):
        """
        Counts occurrences of chords in the dataset.
        """
        for seq in sequences:
            # Update vocabulary
            self.vocab.update(seq)
            
            # Count for every order from 1 to N
            for order in range(1, self.N + 1):
                for i in range(len(seq) - order + 1):
                    ngram = tuple(seq[i : i+order])
                    history = ngram[:-1]
                    target = ngram[-1]
                    
                    self.counts[order][history][target] += 1
                    self.total_counts[order][history] += 1
        
        print(f"Training complete. Vocab size: {len(self.vocab)}")

    def get_ml_prob(self, history, chord):
        """
        Calculates Maximum Likelihood Probability.
        P_ML = count(history + chord) / count(history)
        """
        order = len(history) + 1
        if order > self.N: return 0
        
        hist_count = self.total_counts[order][history]
        if hist_count == 0:
            return 0.0
        
        return self.counts[order][history][chord] / hist_count

    # --- 3. Jelinek-Mercer Smoothing Implementation ---

    def get_jm_prob(self, history, chord, lambdas):
        """
        Recursive Interpolation.
        P_JM(C|History) = lambda * P_ML(C|History) + (1-lambda) * P_JM(C|Shorter_History)
        """
        # Ensure history matches the model order
        history = tuple(history)[-(self.N-1):]
        order = len(history) + 1
        
        # Get the lambda for this specific order (e.g., lambda_2 for Trigrams)
        # Note: lambdas list should be [lambda_0, lambda_1, lambda_2...]
        lam = lambdas[order - 1] if order - 1 < len(lambdas) else 0.5
        
        # 1. Calculate Maximum Likelihood for this level
        p_ml = self.get_ml_prob(history, chord)
        
        # 2. Recursive Step
        if order == 1:
            # Base Case (Order 0): Uniform Distribution (1 / Vocabulary Size)
            # The paper mentions this in Section 2.2
            p_lower = 1.0 / len(self.vocab) if self.vocab else 1e-6
        else:
            # Recursive call with shortened history (Backoff)
            shorter_history = history[1:]
            p_lower = self.get_jm_prob(shorter_history, chord, lambdas)
            
        # 3. Combine
        return (lam * p_ml) + ((1 - lam) * p_lower)

    def predict_next(self, input_sequence, lambdas=None, top_k=5, use_smoothing=True):
        """
        Predicts the most likely next chord(s) given an input sequence.
        
        Args:
            input_sequence: List of chords (e.g., ['C:maj', 'G:maj'])
            lambdas: Lambda values for JM smoothing (default: [0.1, 0.5, 0.9])
            top_k: Number of top predictions to return
            use_smoothing: Whether to use JM smoothing (True) or raw ML (False)
        
        Returns:
            List of tuples: [(chord, probability), ...]
        """
        if lambdas is None:
            lambdas = [0.1, 0.5, 0.9]
        
        # Extract the relevant history (last N-1 chords)
        history = tuple(input_sequence)[-(self.N - 1):]
        
        # Calculate probability for each chord in vocabulary
        predictions = []
        for chord in self.vocab:
            if use_smoothing:
                prob = self.get_jm_prob(history, chord, lambdas)
            else:
                prob = self.get_ml_prob(history, chord)
            predictions.append((chord, prob))
        
        # Sort by probability (descending) and return top_k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def save(self, filepath):
        """Save the trained model to a file using pickle."""
        print(f"Saving model to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print("Model saved successfully.")
    
    @classmethod
    def load(cls, filepath):
        """Load a trained model from a file."""
        print(f"Loading model from {filepath}...")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
        return model

# --- Data Loading Helpers ---

def standardize_chord(chord_label: str) -> str:
    """Convert chord label to standardized format.
    
    Rules:
    - Major chords (X:maj) → X (root only)
    - Minor chords (X:min) → Xm
    - Seventh chords (X:7) → X (treated as major)
    - Half-diminished 7th (X:hdim7) → Xm (treated as minor)
    - Inversions (/N) are ignored
    """
    # Remove inversion if present
    chord_label = chord_label.split("/")[0]
    
    if ":" not in chord_label:
        return chord_label  # Return as-is if malformed
    
    root, quality = chord_label.split(":", 1)
    
    # Map quality to standardized format
    if quality == "maj":
        return root  # Major → just root
    elif quality == "min":
        return f"{root}m"  # Minor → root + "m"
    elif quality == "7":
        return f"{root}7"  # Dominant 7th → major (just root)
    elif quality == "hdim7":
        return f"{root}hdim7"  # Half-diminished → minor
    else:
        # Default: treat unknown as major
        return root

def extract_chord_progression(jam_path: Path) -> List[str]:
    """Extract chord progression from a JAMS file.
    
    Returns a list of chord labels (strings).
    """
    try:
        jam = jams.load(str(jam_path))
        chord_anns = jam.search(namespace="chord")
        
        if not chord_anns:
            return []
        
        return [event.value for event in chord_anns[0].data]
    except Exception as e:
        print(f"Error loading {jam_path}: {e}")
        return []

if __name__ == "__main__":
    # Define paths
    current_dir = Path(__file__).parent
    # Assuming the script is in models/n_gram_chord_progression/
    # and guitarset is in the project root
    project_root = current_dir.parent.parent
    annotation_dir = project_root / 'guitarset' / 'annotation'
    model_path = current_dir / 'ngram_model.pkl'

    print(f"Looking for annotations in: {annotation_dir}")

    if not annotation_dir.exists():
        print("Error: Annotation directory not found!")
    else:
        # Collect chord progressions
        chord_progressions = []
        
        # Get all _comp.jams files
        comp_files = sorted([f for f in annotation_dir.glob("*.jams") if f.stem.endswith("_comp")])
        print(f"Found {len(comp_files)} accompaniment files")
        
        if len(comp_files) == 0:
             # Try recursive search if flat structure is not as expected
             print("Trying recursive search...")
             comp_files = sorted([f for f in annotation_dir.rglob("*.jams") if f.stem.endswith("_comp")])
             print(f"Found {len(comp_files)} accompaniment files recursively")

        for jam_path in comp_files:
            progression = extract_chord_progression(jam_path)
            chord_progressions.append(progression)

        if chord_progressions:
            # Apply standardization
            chord_progressions_std = [
                [standardize_chord(c) for c in prog] 
                for prog in chord_progressions
            ]
            
            print(f"Processed {len(chord_progressions_std)} sequences.")

            # Train model
            N = 3 # Trigram model
            model = NGramModel(n=N)
            model.train(chord_progressions_std)
            
            # Save model
            model.save(model_path)
            
            # Test loading
            loaded_model = NGramModel.load(model_path)
            
            # Simple test
            test_seq = ['C', 'G']
            print(f"Testing prediction for {test_seq}:")
            preds = loaded_model.predict_next(test_seq, top_k=3)
            print(preds)

            print(f'Vocab: {model.vocab}')
            
        else:
            print("No chord progressions found to train on.")
