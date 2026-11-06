# FastSpeech 2 - Telugu (IndicTTS) Speech Synthesis

This module implements FastSpeech 2 for Telugu text-to-speech synthesis, trained on the **IndicTTS Telugu** dataset.

## Overview
- **Dataset**: IndicTTS Telugu
- **Training samples**: 2,613 (~5.00 hours)
- **Validation samples**: 525 (~1.00 hours)
- **Acoustic parameters**: 22,050 Hz sampling rate, 80 mel channels
- **Alignment**: Montreal Forced Aligner (MFA) with Telugu grapheme dictionary
- **Text Frontend**: Unicode grapheme-level tokenization (`0x0C00`–`0x0C7F` + digits)
- **Model Checkpoint**: 40,000 steps (`checkpoints/40000.pth.tar`)
- **Vocoder**: HiFi-GAN Universal Vocoder

## Directory Structure
- `checkpoints/`: Model checkpoint weights (`40000.pth.tar`)
- `config/`: Configuration files (`config/Telugu/`)
- `data_split/`: `Telugu_split.json` dataset partition
- `hifigan/`: HiFi-GAN vocoder implementation and checkpoint
- `inference_stats/`: Pitch and energy normalization statistics (`stats.json`, `speakers.json`)
- `lexicon/`: Pronunciation lexicon dictionary
- `logs/`: Training and validation step logs (`logs/train/log.txt`, `logs/val/log.txt`)
- `model/`: FastSpeech 2 architecture, loss functions, and optimizer
- `output/`: Synthesized audio samples and loss curves
- `preprocessor/`: Feature extraction and MFA alignment parsers
- `text/`: Text processing, Telugu Unicode grapheme mappings, and symbol definitions
- `transformer/`: Feed-Forward Transformer layers and attention blocks
- `utils/`: Model inference and helper routines

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Single Sentence Synthesis
Synthesize speech for a custom Telugu sentence:
```bash
python synthesize.py \
  --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" \
  --restore_step 40000 \
  --mode single \
  -p config/Telugu/preprocess.yaml \
  -m config/Telugu/model.yaml \
  -t config/Telugu/train.yaml
```
Output audio will be saved in `output/result/`.

### 3. Automated Inference & Analysis
Run the complete evaluation suite on the benchmark test sentences:
```bash
python inference.py --out ../Results/Telugu --restore_step 40000
```
This generates:
- Synthesized 22.05 kHz WAV files in `../Results/Telugu/test_generated_audios/`
- Duration, pitch, and energy plots in `../Results/Telugu/variance_adaptor_plots/`
- Training and validation loss curves in `../Results/Telugu/`

### 4. Prosody Controllability
Adjust speaking speed, pitch, and loudness:
```bash
python synthesize.py \
  --text "ఈ రోజు చాలా మంచి రోజు." \
  --restore_step 40000 \
  --mode single \
  -p config/Telugu/preprocess.yaml \
  -m config/Telugu/model.yaml \
  -t config/Telugu/train.yaml \
  --duration_control 0.9 \
  --pitch_control 1.0 \
  --energy_control 1.1
```
