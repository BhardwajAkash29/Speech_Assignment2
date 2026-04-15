# 🎙️ Speech PA-2: Code-Switched (Hinglish) Transcription → Low-Resource Language Voice Cloning

> **Programming Assignment 2 — Speech Understanding Course**  
> End-to-end pipeline: Hinglish lecture audio → denoising → LID → ASR → IPA → Santhali translation → zero-shot TTS → adversarial robustness evaluation

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Running the Notebook](#running-the-notebook)
- [Module Breakdown](#module-breakdown)
  - [Part 0: Setup & Data](#part-0-setup--data)
  - [Part I: ASR & Language Identification](#part-i-asr--language-identification)
  - [Part II: IPA Conversion & Translation](#part-ii-ipa-conversion--translation)
  - [Part III: Voice Cloning & TTS Synthesis](#part-iii-voice-cloning--tts-synthesis)
  - [Part IV: Anti-Spoofing & Adversarial Attacks](#part-iv-anti-spoofing--adversarial-attacks)
  - [Part V: Evaluation & Submission](#part-v-evaluation--submission)
- [Evaluation Metrics & Targets](#evaluation-metrics--targets)
- [Output Files](#output-files)
- [Known Issues & Fixes](#known-issues--fixes)
- [References](#references)

---

## Overview

This assignment implements a complete speech processing pipeline that:

1. Takes a **Hinglish (Hindi + English code-switched)** lecture audio recording
2. Denoises it using spectral gating (`noisereduce`)
3. Performs **frame-level Language Identification (LID)** using a BiLSTM + multi-head attention model on Wav2Vec2 features
4. Transcribes speech using **OpenAI Whisper large-v3** with N-gram language model biasing
5. Converts the transcript to **IPA phoneme notation** (handling Hindi retroflex/aspirate/breathy consonants)
6. **Translates** the transcript to **Santhali** (a low-resource Scheduled Tribal language of India) using a custom 500-term parallel corpus in Ol Chiki script
7. Synthesizes **zero-shot voice-cloned speech** in Santhali using Kokoro TTS + DTW-based prosody warping
8. Evaluates **anti-spoofing** robustness with an LFCC + MLP classifier
9. Demonstrates **FGSM adversarial attacks** on the LID model

---

## Pipeline Architecture

```
Raw Lecture Audio (.wav)
        │
        ▼
┌─────────────────────┐
│  Cell 1.1: Denoising│  ← noisereduce (spectral gating)
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Cell 1.2: Frame-Level LID  │  ← Wav2Vec2 + BiLSTM + MultiHeadAttention
│  (Hindi / English labels)   │    F1 target ≥ 0.85
└────────┬────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Cell 1.3: Whisper ASR             │  ← Whisper large-v3 + KN trigram LM
│  + N-gram Logit Bias               │    WER(EN)<15%, WER(HI)<25%
└────────┬───────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Cell 2.1: Hinglish → IPA    │  ← Custom G2P for Devanagari + en-us espeak
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Cell 2.2: EN → Santhali (LRL)  │  ← 500-term parallel corpus (Ol Chiki)
└────────┬────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Cell 3.1: d-Vector Speaker Embed.   │  ← 3-layer LSTM d-vector (256-d)
└────────┬─────────────────────────────┘
         │
         ▼
┌───────────────────────────────────┐
│  Cell 3.2: DTW Prosody Warping    │  ← Parselmouth F0 + fastdtw alignment
└────────┬──────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Cell 3.3: Zero-Shot TTS (Kokoro)    │  ← Kokoro neural TTS + pitch shift
│  Output: Santhali synthesized audio  │    MCD target < 8.0 dB
└────────┬─────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────┐
│  Cell 4.1: Anti-Spoofing CM (LFCC)    │  ← LFCC + MLP, EER target < 10%
└────────┬──────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Cell 4.2: FGSM Adversarial Attack on LID  │  ← ε sweep, SNR > 40 dB target
└────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Cell 5: Eval Summary    │
│  Cell 6: Package & ZIP   │
└──────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ (Google Colab default) |
| CUDA | 12.4 (T4 GPU recommended) |
| Google Colab | GPU runtime required |
| RAM | ≥ 12 GB (T4 provides 15 GB) |
| Disk | ≥ 10 GB free in `/content` |

> ⚠️ **This notebook is designed to run on Google Colab with a T4 GPU.**  
> Set: `Runtime → Change runtime type → GPU (T4)`

---

## Setup & Installation

All dependencies are installed in **Cell 0.1**. Run cells strictly in order — each cell depends on variables from previous ones.

Key packages installed:

```bash
# Core numerics (pinned for ABI compatibility)
numpy>=2.0.0,<2.1.0
scipy librosa soundfile

# Denoising
noisereduce

# PyTorch for CUDA 12.4
torch torchaudio  --index-url https://download.pytorch.org/whl/cu124

# ASR & Transformers
transformers>=4.40 accelerate openai-whisper

# Audio analysis
praat-parselmouth pydub

# Phonemization / IPA
epitran phonemizer gruut indic-transliteration

# TTS
kokoro  (Python 3.12 compatible; replaces Coqui YourTTS)

# Evaluation & ML
dtw-python fastdtw scikit-learn jiwer
matplotlib seaborn pandas gdown
```

> After Cell 0.1 completes, **restart the Colab runtime** before proceeding (as instructed in the cell output).

---

## Dataset

The lecture audio is automatically downloaded from Google Drive in **Cell 0.2**.

- **Source:** Google Drive folder (shared link in `DRIVE_FOLDER_URL`)
- **Expected content:** One or more audio files (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`)
- **Saved to:** `/content/speech_data/`

The first audio file (alphabetically) is selected as the primary lecture audio. A 10-minute segment (0–600 s) is extracted for all downstream processing.

If you want to use your own audio, place a `.wav` file at `/content/speech_data/your_audio.wav` before running Cell 0.2.

---

## Running the Notebook

Run all cells **in order**, top to bottom. Each cell declares the globals it needs from prior cells.

```
Cell 0.1 → Install deps           (restart runtime after this)
Cell 0.2 → Download dataset
Cell 0.3 → Audio inspection + 10-min segment extraction
Cell 1.1 → Denoising
Cell 1.2 → Frame-level LID (train + evaluate)
Cell 1.3 → Whisper ASR + N-gram LM
Cell 1.4 → WER + LID timestamp evaluation
Cell 2.1 → Hinglish → IPA
Cell 2.2 → EN → Santhali translation
Cell 3.1 → Upload reference voice + d-vector extraction
Cell 3.2 → F0 extraction + DTW prosody warping
Cell 3.3 → Zero-shot TTS synthesis (Kokoro)
Cell 3.4 → MCD evaluation
Cell 4.1 → Anti-spoofing classifier (LFCC + MLP)
Cell 4.2 → FGSM adversarial attack
Cell 5   → Full evaluation summary + confusion matrix
Cell 6   → Package outputs + download ZIP
```

> 📌 **Cell 3.1** prompts you to upload a 60-second personal voice recording. If you skip the upload, a synthetic 60 s sinusoidal reference is used automatically.

> 📌 **Cell 6** — change `ROLLNO = "YourRollNo"` to your actual roll number before running.

---

## Module Breakdown

### Part 0: Setup & Data

| Cell | Task | Key Output |
|------|------|------------|
| 0.1 | Install all Python dependencies | Stable environment |
| 0.2 | Download lecture audio from Google Drive | `/content/speech_data/` |
| 0.3 | Load audio, extract 10-min segment, plot waveform | `original_segment.wav`, `waveform_original.png` |

---

### Part I: ASR & Language Identification

#### Cell 1.1 — Denoising (Task 1.3)
- **Method:** `noisereduce` spectral gating (replaces DeepFilterNet which requires Rust compilation unavailable in Colab)
- **Noise profile:** First 0.5 s of audio used as noise reference
- **Output:** `denoised_segment.wav`
- **Metric:** SNR improvement reported in dB

#### Cell 1.2 — Multi-Head Frame-Level LID (Task 1.1)
- **Backbone:** `facebook/wav2vec2-base` (frozen feature extractor)
- **LID Head:** `Linear(768→256) → BiLSTM(2-layer) → MultiHeadAttention(4 heads) → LayerNorm → Classifier(2 classes)`
- **Labels:** 0 = Hindi, 1 = English
- **Inference:** Sliding window (2 s window, 1 s hop)
- **Saved:** `lid_weights.pt`, `lid_results` global (list of timestamp dicts)
- **Target:** F1 (macro) ≥ 0.85

#### Cell 1.3 — N-gram LM + Constrained Decoding (Task 1.2)
- **ASR Model:** OpenAI Whisper `large-v3`
- **LM:** Kneser-Ney smoothed trigram LM trained on a speech/NLP domain corpus (~60 technical terms)
- **Bias method:** Domain vocab injected via Whisper's `initial_prompt` parameter; `condition_on_previous_text=True`
- **Output:** `transcript.txt`

#### Cell 1.4 — WER Evaluation & LID Timestamp Alignment
- **WER** computed separately for English and Hindi sub-sequences using `jiwer`
- **Timestamp precision** measured as mean boundary error (ms) between LID predictions and Whisper word timestamps
- **Targets:** WER(EN) < 15%, WER(HI) < 25%, timestamp error < 200 ms

---

### Part II: IPA Conversion & Translation

#### Cell 2.1 — Hinglish → IPA (Task 2.1)
- Devanagari text: ITRANS transliteration via `indic-transliteration`, then rule-based Hindi → IPA mapping
- English text: `phonemizer` with `espeak` backend (`en-us`), `njobs=1` for Colab stability
- Technical terms: Custom dictionary of 30+ speech/NLP terms with hand-verified IPA
- Special handling: retroflex (ट/ड → ʈ/ɖ), aspirates (ख/घ → kʰ/ɡʱ), breathy (ह → ɦ)
- **Output:** `ipa_transcript.txt`

#### Cell 2.2 — Translation to Santhali (Task 2.2)
- **Target language:** Santhali (Santali) — ~7 million speakers, Ol Chiki script
- **Corpus:** 500 manually curated English↔Santhali term pairs covering speech processing, linguistics, and academic vocabulary
- Both Ol Chiki (ᱵᱳᱞᱤ) and Roman transliterations provided
- **Outputs:** `santhali_transcript.txt`, `parallel_corpus.tsv`

---

### Part III: Voice Cloning & TTS Synthesis

#### Cell 3.1 — Voice Recording & d-Vector Extraction (Task 3.1)
- Upload a 60 s reference WAV of your voice; if skipped, a synthetic 60 s reference is generated
- **d-Vector network:** `MFCC(40-dim) → 3×LSTM(256) → Linear(256) → L2-normalize`
- Sliding window extraction (1.5 s windows, 0.75 s hop), embeddings averaged
- **Output:** `speaker_embedding.npy` (256-d L2-normalized vector)

#### Cell 3.2 — Prosody Warping with DTW (Task 3.2)
- **F0 extraction:** Parselmouth/Praat (`pitch_floor=75 Hz`, `pitch_ceiling=500 Hz`)
- **DTW alignment:** `dtw-python` aligns log-F0 contours of source (professor) and reference (student)
- Warped F0 smoothed with Gaussian filter (σ=3 frames)
- **Output:** `warped_f0.npy`, `source_energy.npy`, `prosody_warping.png`

#### Cell 3.3 — Zero-Shot TTS Synthesis (Task 3.3)
- **TTS Engine:** [Kokoro](https://github.com/hexgrad/kokoro) (Python 3.12 compatible; replaces Coqui YourTTS which requires Python < 3.12)
- Santhali transcript chunked into ≤ 200-character segments
- Prosody-aware pitch shift: F0 ratio (source/reference) → semitone shift via `librosa.effects.pitch_shift`
- **Output:** `output_LRL_cloned.wav` (~24 kHz)
- **Target:** MCD < 8.0 dB

#### Cell 3.4 — MCD Evaluation
- **Mel Cepstral Distortion** computed using DTW-aligned 13-dimensional MCEPs (C1–C13, excluding C0)
- Formula: `MCD = (10/ln10) × √(2 × mean_path‖mc_ref − mc_syn‖²)`
- Ablation: estimated MCD without DTW prosody warping shown for comparison

---

### Part IV: Anti-Spoofing & Adversarial Attacks

#### Cell 4.1 — Anti-Spoofing Classifier (Task 4.1)
- **Features:** LFCC (Linear Frequency Cepstral Coefficients, 60-dim) with linear filterbank — 120-d feature vector (mean + std over time)
- **Model:** MLP with architecture `120 → 256 → 128 → 64 → 2` with BatchNorm + Dropout(0.3)
- **Training:** Adam optimizer, 50 epochs, bona-fide vs synthesized audio
- **Evaluation:** Equal Error Rate (EER) on 30% hold-out split
- **Output:** `cm_weights.pt`, `roc_antispoof.png`
- **Target:** EER < 10%

#### Cell 4.2 — FGSM Adversarial Attack (Task 4.2)
- **Attack:** Fast Gradient Sign Method on the LID model
- `x_adv = x + ε × sign(∇_x L(f(x), y_target))`
- Target class: English (fool model into misclassifying Hindi as English)
- **ε sweep:** `[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]`
- **Viability criterion:** Attack flips prediction AND SNR > 40 dB (perturbation inaudible)
- **Output:** `fgsm_epsilon_snr.png`, minimum viable ε reported
- **Important fix:** Uses `torch.enable_grad()` context to ensure gradients flow through the Wav2Vec2 + LID forward pass correctly

---

### Part V: Evaluation & Submission

#### Cell 5 — Full Evaluation Summary
Prints a consolidated metrics table, renders the LID confusion matrix, and lists the submission file manifest with sizes.

#### Cell 6 — Package & Download
Copies all output files into a structured directory and creates a downloadable ZIP:

```
PA2_submission/
├── audio/
│   ├── original_segment.wav
│   ├── student_voice_ref.wav
│   ├── denoised_segment.wav
│   └── output_LRL_cloned.wav
├── text/
│   ├── transcript_hinglish.txt
│   ├── ipa_transcript.txt
│   ├── santhali_transcript.txt
│   └── parallel_corpus.tsv
├── models/
│   ├── lid_weights.pt
│   ├── speaker_embedding.npy
│   └── cm_weights.pt
├── figures/
│   ├── waveform_original.png
│   ├── prosody_warping.png
│   ├── roc_antispoof.png
│   ├── fgsm_epsilon_snr.png
│   └── lid_confusion_matrix.png
└── README.md
```

> ⚠️ Set `ROLLNO = "YourRollNo"` before running Cell 6.

---

## Evaluation Metrics & Targets

| Task | Metric | Target | Notes |
|------|--------|--------|-------|
| 1.1 LID | F1 (macro) | ≥ 0.85 | Hindi / English frame-level |
| 1.1 LID | Timestamp boundary error | < 200 ms | vs Whisper word-level timestamps |
| 1.2 ASR | WER (English) | < 15% | Whisper + N-gram bias |
| 1.2 ASR | WER (Hindi) | < 25% | Whisper + N-gram bias |
| 1.3 Denoising | SNR improvement | Positive | noisereduce spectral gating |
| 3.3 TTS | MCD | < 8.0 dB | DTW-aligned MCEPs |
| 4.1 CM | EER | < 10% | LFCC + MLP anti-spoofing |
| 4.2 Adv. | Min viable ε | SNR > 40 dB | FGSM inaudible attack |

---

## Output Files

| File | Description |
|------|-------------|
| `audio/original_segment.wav` | 10-minute raw lecture segment (16 kHz) |
| `audio/denoised_segment.wav` | Denoised version |
| `audio/student_voice_ref.wav` | Uploaded student reference voice (60 s) |
| `audio/output_LRL_cloned.wav` | Synthesized Santhali audio (24 kHz) |
| `text/transcript_hinglish.txt` | Full Whisper transcript |
| `text/ipa_transcript.txt` | IPA phoneme representation |
| `text/santhali_transcript.txt` | Santhali (Ol Chiki) translation |
| `text/parallel_corpus.tsv` | 500-term EN ↔ Santhali bilingual lexicon |
| `models/lid_weights.pt` | Trained LID model weights |
| `models/speaker_embedding.npy` | 256-d d-vector embedding |
| `models/cm_weights.pt` | Trained anti-spoofing CM weights |
| `figures/waveform_original.png` | Waveform plot of 10-min segment |
| `figures/prosody_warping.png` | F0 contour comparison (source / ref / warped) |
| `figures/roc_antispoof.png` | Anti-spoofing ROC curve with EER marker |
| `figures/fgsm_epsilon_snr.png` | Epsilon vs SNR plot for adversarial sweep |
| `figures/lid_confusion_matrix.png` | Hindi / English LID confusion matrix |

---

## Known Issues & Fixes

This notebook contains **14 documented bug fixes** from the original submission:

| Cell | Bug | Fix Applied |
|------|-----|-------------|
| 0.1 | Missing `fastdtw` install caused NameError in Cell 3.4 | Moved to Cell 0.1 |
| 0.2 | `gdown` failed on shared folder URL | Added `fuzzy=True` flag |
| 0.3 | `librosa.display` not imported | Explicit `import librosa.display` |
| 1.1 | DeepFilterNet requires Rust compiler (unavailable in Colab) | Replaced with `noisereduce` |
| 1.2 | `device` undefined in cell scope | Added global `device` declaration |
| 1.2 | `Wav2Vec2FeatureExtractor` vs `Wav2Vec2Processor` inconsistency | Unified to `Wav2Vec2Processor` |
| 1.2 | `lid_results`, `predicted`, `ground_truth`, `f1` not exported as globals | Saved as module-level globals |
| 1.3 | `NgramLM.train()` body truncated in uploaded version | Fully restored |
| 1.4 | `mean_err` used before assignment when `errors_ms` is empty | Wrapped in `if errors_ms:` guard |
| 3.1 | `files.upload()` MessageError not caught defensively | Broader try/except with synthetic fallback |
| 3.2 | `dtw()` treated as function; it is a class | Use `DTW(...)` instance with `.distance`/`.index1` attributes |
| 3.3 | `tts.languages` attribute raises AttributeError | Wrapped in `getattr` |
| 3.3 | `OUTPUT_SYNTHESIS_PATH` defined inside conditional block | Moved to cell top-level |
| 4.2 | `requires_grad_(True)` inside `torch.no_grad()` silently does nothing | Replaced with `torch.enable_grad()` scope |

---

## References

- Radford, A. et al. (2023). **Robust Speech Recognition via Large-Scale Weak Supervision** (Whisper). ICML.
- Baevski, A. et al. (2020). **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**. NeurIPS.
- Casanova, E. et al. (2022). **YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion**. ICML.
- Wan, L. et al. (2018). **Generalized End-to-End Loss for Speaker Verification** (d-vector). ICASSP.
- Goodfellow, I. et al. (2015). **Explaining and Harnessing Adversarial Examples** (FGSM). ICLR.
- Todisco, M. et al. (2019). **ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection**. Interspeech.
- Müller, M. (2007). **Dynamic Time Warping**. Information Retrieval for Music and Motion.
- kokoro TTS: https://github.com/hexgrad/kokoro
- noisereduce: https://github.com/timsainb/noisereduce

---

*Last updated: April 2026*
