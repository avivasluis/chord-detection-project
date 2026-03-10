# Automatic Guitar Chord Recognition

**A Comparative Study of Deep Learning and Template-Based Methods**

This project processes the [GuitarSet](https://guitarset.weebly.com/) dataset to build a high-quality 24-class chord recognition pipeline. We train an AlexNet-based classifier on CQT chromagrams and achieve strong results (~80.7% accuracy) while comparing against a classical pattern-matching baseline (~60.2%). The trained model powers two practical use cases: **offline chord annotation** of audio files and **real-time chord detection** from a microphone or system audio.

---

## Results

| Method | Accuracy | F1 (Macro) |
|--------|----------|------------|
| **Pattern matching** (template-based) | 60.24% | 0.578 |
| **AlexNet + CQT chromagram** (best, v10) | **80.71%** | **0.789** |

The pattern-matching baseline uses music-theory chord templates (major/minor), `chroma_stft`, KL-divergence scoring, and median filtering. It achieves a respectable result, which **confirms that chromagrams are a meaningful representation** for chord recognition. The CNN further learns discriminative patterns from CQT chromagrams and improves accuracy by ~20 percentage points.
