# 🚀 Deploy Your App - Step-by-Step

## ✅ Pre-Deployment Checklist

Before deploying, verify everything:

```
✓ App running locally? http://localhost:8501
✓ .env file created? Check: ls -la .env
✓ API keys valid? HF_TOKEN and SARVAM_API_KEY
✓ .env in .gitignore? Should not push to GitHub
✓ All tests passed? Ran Live and Research modes
```

---

## 🚀 Option 1: Streamlit Cloud (EASIEST - 5 minutes)

### Step 1: Push to GitHub
```bash
cd C:\Hindi_Live_Streamlit_App\hindi_live_streamlit

git init
git add .
git commit -m "Hindi ASR TTS app with research evaluation"
git remote add origin https://github.com/YOUR_USERNAME/hindi-asr-tts.git
git branch -M main
git push -u origin main
```

### Step 2: Sign Up
1. Visit https://streamlit.io/cloud
2. Click "Sign Up"
3. Use GitHub account
4. Authorize Streamlit

### Step 3: Deploy
1. Click "Create app"
2. Select your repo: `hindi-asr-tts`
3. Branch: `main`
4. File: `app.py`
5. Click "Deploy"

### Step 4: Add Secrets
1. App will load (shows error about missing keys)
2. Click "⋯" (menu) → "Settings"
3. Click "Secrets"
4. Paste this:
```
HF_TOKEN = "hf_your_actual_token_here"
SARVAM_API_KEY = "sk_your_actual_key_here"
```
5. Save → App auto-restarts ✅

### Step 5: Your Live App!
- **URL**: https://your-username-hindi-asr-tts.streamlit.app
- **Share**: Send link to anyone
- **Updates**: Push to GitHub, app auto-deploys

---

## 🐳 Option 2: Docker (5 minutes)

### Step 1: Build
```bash
docker build -t hindi-asr-tts .
```

### Step 2: Run Locally (Test)
```bash
docker run -e HF_TOKEN="hf_your_token" \
           -e SARVAM_API_KEY="sk_your_key" \
           -p 8501:8501 \
           hindi-asr-tts
```

Visit: http://localhost:8501

### Step 3: Deploy to Server
**Push to Docker Hub:**
```bash
docker login
docker tag hindi-asr-tts your-username/hindi-asr-tts
docker push your-username/hindi-asr-tts
```

**On your server (AWS EC2, DigitalOcean, etc):**
```bash
docker run -e HF_TOKEN="your_token" \
           -e SARVAM_API_KEY="your_key" \
           -p 80:8501 \
           your-username/hindi-asr-tts
```

Your app is live at: `http://your-server-ip`

---

## 🖥️ Option 3: Self-Hosted (10 minutes)

### Step 1: SSH into Server
```bash
ssh user@your-server.com
```

### Step 2: Clone & Setup
```bash
git clone https://github.com/your-username/hindi-asr-tts.git
cd hindi-asr-tts

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Create .env
```bash
cat > .env << 'EOF'
HF_TOKEN=hf_your_actual_token
SARVAM_API_KEY=sk_your_actual_key
