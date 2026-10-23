# Hindi Live ASR → TTS

A Streamlit app for real-time Hindi speech recognition and synthesis.

**Features:**
- 🎤 **Live Mode**: Record or upload Hindi audio → Transcribe → Synthesize speech
- 🔬 **Research Mode**: Compare ASR output against ground truth (WER/CER metrics)
- 📊 **Analysis**: Audio metrics, spectrograms, real-time factors

**Models:**
- ASR: [Koyal Hindi 120M](https://huggingface.co/adalat-ai/koyal-hi-120m-1.0)
- TTS: [Sarvam Bulbul v3](https://console.sarvam.ai/)

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd hindi_live_streamlit
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get API Keys

**Hugging Face Token:**
- Go to https://huggingface.co/settings/tokens
- Create a **Read** token (select "repo.content.read")
- Copy the token

**Sarvam API Key:**
- Go to https://console.sarvam.ai/
- Create an API key
- Copy the key

### 3. Configure Secrets

```bash
cp .env.example .env
```

Edit `.env`:
```
HF_TOKEN=hf_xxxxxxxxxxxx
SARVAM_API_KEY=sk_xxxxxxxxxxxx
```

**⚠️ NEVER commit `.env` to git!** It's in `.gitignore` for your safety.

### 4. Run

```bash
streamlit run app.py
```

Visit http://localhost:8501

---

## 📋 Usage

### Live Mode
1. Record Hindi speech or upload an audio file (WAV, MP3, M4A, etc.)
2. Click **"Transcribe and speak"**
3. View transcript and listen to generated speech
4. See performance metrics (latency, real-time factor)

### Research Mode
1. Upload a `.txt` file with the ground-truth transcript
2. Record/upload audio
3. Click **"Transcribe and speak"**
4. View WER/CER metrics + detailed audio analysis

---

## 🔧 Requirements

- Python 3.10+
- 16GB+ RAM (recommended)
- NVIDIA GPU (optional, for faster processing)

**Note:** The Koyal model runs on CPU by default. For low-memory systems, the app uses float16 precision to reduce memory footprint.

---

## 📁 Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Template for API keys
├── .gitignore          # Excludes .env and secrets
├── README.md           # This file
└── SECURITY.md         # Security best practices
```

---

## 🔐 Security

**Important:**
- ✅ API keys are stored in `.env` (not committed)
- ✅ Secrets use Streamlit's encrypted storage
- ✅ Input validation on all uploads
- ✅ Error messages don't leak sensitive data
- ✅ No logs contain API keys

See [SECURITY.md](SECURITY.md) for deployment guides (Streamlit Cloud, Docker, self-hosted).

---

## 🐛 Troubleshooting

### "HF_TOKEN is missing"
- Ensure `.env` exists with your Hugging Face token
- Or set environment variable: `export HF_TOKEN=your_token`

### "SARVAM_API_KEY is missing"
- Ensure `.env` has your Sarvam API key
- Check that your API key is active on https://console.sarvam.ai/

### `No module named 'torchvision'` in Streamlit logs
- This can be a file-watcher warning while Streamlit inspects Transformers' optional image-processing modules; audio transcription does not require `torchvision`.
- The app disables the file watcher to avoid this warning. Automatic app reloads after code changes are also disabled, so restart Streamlit after editing files.

### Memory error during transcription
- Close other applications to free RAM
- The app uses float16 precision to minimize memory usage
- Consider using a GPU or cloud deployment (Streamlit Cloud)
- SraVaani is a large local model; avoid selecting it on hosts with limited memory. Its load errors are shown in Research mode's ASR comparison and run log.

### SraVaani fails to load in Research mode
- Install the pinned Transformers 4.x dependency from `requirements.txt` (`pip install -r requirements.txt`) and restart the app.
- SraVaani uses custom Hugging Face model code pinned to a fixed repository revision. Confirm your Hugging Face token has access if the model download is denied.
- The other Hugging Face ASR/TTS checkpoints are pinned to fixed revisions in `research.py`; update those revisions deliberately when upgrading a model.

### Slow processing
- Recording directly is faster than uploading files
- First run downloads the Koyal model (~1.5 GB) — this is cached afterwards
- Real-time factor ~0.5x means 2s audio takes ~1s to transcribe

---

## 📊 Metrics Explained

### Performance
- **ASR RTF**: Real-Time Factor = (processing time) / (audio duration)
  - RTF < 1.0 = faster than real-time
- **TTS RTF**: Same metric for speech synthesis
- **WER**: Word Error Rate — % of words incorrect (Research mode)
- **CER**: Character Error Rate — % of characters incorrect (Research mode)

### Audio Analysis
- **Spectral Centroid**: Average frequency (Hz)
- **Zero-Crossing Rate**: Measure of noisiness
- **RMS Level**: Root Mean Square amplitude
- **Peak Level**: Loudest point (dBFS)

---

## 🚢 Deployment

### Streamlit Cloud (Easiest)
1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repo
4. Add secrets in **Settings → Secrets**:
   ```toml
   HF_TOKEN = "your_token"
   SARVAM_API_KEY = "your_key"
   ```

### Docker
```bash
docker build -t hindi-asr-tts .
docker run -e HF_TOKEN=... -e SARVAM_API_KEY=... -p 8501:8501 hindi-asr-tts
```

See [SECURITY.md](SECURITY.md) for full deployment guides.

---

## 📝 License

[Add your license here]

---

## 🙏 Credits

- **Koyal ASR**: [adalat-ai/koyal-hi-120m-1.0](https://huggingface.co/adalat-ai/koyal-hi-120m-1.0)
- **Sarvam TTS**: [Sarvam AI](https://sarvam.ai/)
- **Streamlit**: [streamlit.io](https://streamlit.io)

---

## 📧 Support

For issues or questions:
1. Check [SECURITY.md](SECURITY.md) for common issues
2. Review error messages carefully
3. Ensure API keys are valid and have available credits
