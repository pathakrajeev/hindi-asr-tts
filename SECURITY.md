# Security Guide

## 🔐 API Keys & Secrets Management

### Local Development
1. **Never commit `.env` files** — Already in `.gitignore`
2. **Keep API keys private**:
   - `HF_TOKEN` — Hugging Face authentication
   - `SARVAM_API_KEY` — Sarvam TTS API key

### For Deployment (Streamlit Cloud / Self-hosted)

#### Streamlit Cloud
Add secrets in **Settings → Secrets** on your Streamlit app dashboard:
```toml
HF_TOKEN = "your_huggingface_token"
SARVAM_API_KEY = "your_sarvam_api_key"
```

#### Self-hosted / Docker
Set environment variables:
```bash
export HF_TOKEN="your_token"
export SARVAM_API_KEY="your_api_key"
streamlit run app.py
```

Or create `.streamlit/secrets.toml` (git-ignored):
```toml
HF_TOKEN = "your_token"
SARVAM_API_KEY = "your_api_key"
```

## 🛡️ Input Validation

The app validates:
- ✅ Audio file size (max 100 MB)
- ✅ Audio duration (max 30 seconds)
- ✅ Transcript length (max 1,200 characters for TTS)
- ✅ Reference text encoding (UTF-8)

## 🚨 Error Handling

**What's NOT leaked:**
- API keys in error messages
- Full API response bodies
- Stack traces with sensitive data

**User-friendly errors:**
- "TTS authentication failed" (not "invalid API key in header")
- "Network error" (not full HTTP response)
- "Invalid audio file" (not librosa stack trace)

## 📋 Before uploading to repo

- [ ] Ensure `.env` is in `.gitignore` ✅
- [ ] Ensure `.streamlit/secrets.toml` is in `.gitignore` ✅
- [ ] Do NOT commit any `.env.example` with real keys
- [ ] Create `.env.example` with placeholder values (see below)
- [ ] Add `SECURITY.md` to repo (this file)
- [ ] Update `README.md` with setup instructions

## 📝 Create `.env.example`

```bash
# Add this file to repo (with placeholder values)
HF_TOKEN=your_huggingface_token_here
SARVAM_API_KEY=your_sarvam_api_key_here
```

Users clone the repo and run:
```bash
cp .env.example .env
# Then fill in their own keys
```

## 🔑 Getting API Keys

### HF_TOKEN
1. Go to https://huggingface.co/settings/tokens
2. Create a **Read-only** token
3. Copy into `.env`

### SARVAM_API_KEY
1. Go to https://console.sarvam.ai/
2. Create an API key
3. Copy into `.env`

## ✅ Security Checklist for Production

- [ ] Use HTTPS only (Streamlit Cloud does this by default)
- [ ] Rotate API keys periodically
- [ ] Monitor API usage for unauthorized access
- [ ] Set API key rate limits / IP whitelisting
- [ ] Use read-only tokens where possible (HF_TOKEN)
- [ ] Never log full API responses
- [ ] Validate all file uploads
- [ ] Set reasonable timeouts (180s for TTS)

## 🚀 Deployment

### Streamlit Cloud (Recommended)
```bash
git push origin main
# Connect repo in Streamlit Cloud dashboard
# Add secrets in Settings → Secrets
```

### Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

Environment variables passed at runtime:
```bash
docker run -e HF_TOKEN="..." -e SARVAM_API_KEY="..." my-app
```

## 📚 References

- [Streamlit Secrets](https://docs.streamlit.io/develop/api-reference/widgets/st.secrets)
- [HF Token Security](https://huggingface.co/docs/hub/security)
- [OWASP: Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
