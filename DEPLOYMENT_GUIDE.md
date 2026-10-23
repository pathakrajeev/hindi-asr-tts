# 🚀 Deployment Guide

## Quick Overview

You have 3 main deployment options:

| Option | Ease | Cost | Setup Time | Best For |
|--------|------|------|-----------|----------|
| **Streamlit Cloud** | ⭐⭐⭐⭐⭐ (Easiest) | Free tier available | 5 min | Most users |
| **Docker** | ⭐⭐⭐ (Medium) | $5-20/month | 10 min | Production servers |
| **Self-hosted** | ⭐⭐ (Hard) | $10-100/month | 30 min | Full control |

---

## Option 1: Streamlit Cloud (Recommended) ⭐⭐⭐⭐⭐

### Easiest & Fastest (5 minutes)

#### Step 1: Push to GitHub
```bash
cd c:\Hindi_Live_Streamlit_App\hindi_live_streamlit

# If not a git repo yet
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Hindi ASR TTS Streamlit app"

# Add remote (replace with YOUR repo)
git remote add origin https://github.com/YOUR_USERNAME/hindi-asr-tts.git

# Push
git branch -M main
git push -u origin main
```

#### Step 2: Create Streamlit Account
1. Go to https://streamlit.io/cloud
2. Click "Sign up"
3. Sign in with GitHub
4. Authorize Streamlit to access your repos

#### Step 3: Deploy Your App
1. Click "New app" → "From existing repo"
2. Select:
   - Repository: `your-username/hindi-asr-tts`
   - Branch: `main`
   - File: `app.py`
3. Click "Deploy"

#### Step 4: Add API Keys (IMPORTANT!)
1. Go to your app dashboard
2. Click "⋯" → "Settings"
3. Go to "Secrets"
4. Paste this (replace with YOUR keys):
```toml
HF_TOKEN = "hf_your_actual_token_here"
SARVAM_API_KEY = "sk_your_actual_key_here"
```
5. Save and app auto-reloads

#### Step 5: Share Your Link
- Your app is live at: `https://your-username-hindi-asr-tts.streamlit.app`
- Share this link with anyone!

---

## Option 2: Docker 🐳

### For Production Servers

#### Step 1: Create Dockerfile
Create file `Dockerfile` in your project root:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create .streamlit config
RUN mkdir -p ~/.streamlit && \
    echo "[server]\nheadless = true\nport = 8501\nenableCORS = false" > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py"]
```

#### Step 2: Create .dockerignore
```
.git
.gitignore
.env
.streamlit/secrets.toml
__pycache__
.venv
*.pyc
.DS_Store
.vscode
Hindi_ASR_TTS_Research_Decision_Analysis*.ipynb
```

#### Step 3: Build Docker Image
```bash
docker build -t hindi-asr-tts:latest .
```

#### Step 4: Run Locally (Test)
```bash
docker run -e HF_TOKEN="hf_your_token" \
           -e SARVAM_API_KEY="sk_your_key" \
           -p 8501:8501 \
           hindi-asr-tts:latest
```

Visit: http://localhost:8501

#### Step 5: Deploy to Server
**Option A: AWS EC2**
```bash
# Push to Docker Hub
docker login
docker tag hindi-asr-tts:latest your-username/hindi-asr-tts:latest
docker push your-username/hindi-asr-tts:latest

# On EC2 instance
ssh ec2-user@your-instance.compute.amazonaws.com
docker pull your-username/hindi-asr-tts:latest
docker run -e HF_TOKEN="..." -e SARVAM_API_KEY="..." -p 80:8501 hindi-asr-tts:latest
```

**Option B: DigitalOcean**
```bash
# Create droplet with Docker pre-installed
# SSH into droplet
ssh root@your-droplet-ip

# Run container
docker run -e HF_TOKEN="..." -e SARVAM_API_KEY="..." -p 80:8501 hindi-asr-tts:latest
```

**Option C: Docker Compose (Multiple containers)**
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      HF_TOKEN: ${HF_TOKEN}
      SARVAM_API_KEY: ${SARVAM_API_KEY}
    volumes:
      - ./.streamlit:/root/.streamlit
```

Run:
```bash
HF_TOKEN="your_token" SARVAM_API_KEY="your_key" docker-compose up
```

---

## Option 3: Self-Hosted Server

### For Advanced Users

#### Step 1: Get a Server
- AWS EC2: $0-20/month
- DigitalOcean: $4-12/month
- Linode: $5-30/month
- Heroku: Free tier (limited)

#### Step 2: Set Up Server
```bash
# SSH into your server
ssh user@your-server-ip

# Install Python
sudo apt-get update
sudo apt-get install python3.11 python3-pip

# Clone repo
git clone https://github.com/your-username/hindi-asr-tts.git
cd hindi-asr-tts

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR on Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Create .env File
```bash
# On server, create .env
nano .env
```

Paste:
```
HF_TOKEN=hf_your_actual_token
SARVAM_API_KEY=sk_your_actual_key
```

Save: Ctrl+O, Enter, Ctrl+X

#### Step 4: Run Streamlit
```bash
streamlit run app.py --server.port 8501
```

#### Step 5: Keep It Running (Use systemd or supervisor)

**Option A: systemd (Recommended)**
Create `/etc/systemd/system/streamlit.service`:
```ini
[Unit]
Description=Streamlit Hindi ASR TTS
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/hindi-asr-tts
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/streamlit run app.py --server.port 8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl start streamlit
sudo systemctl enable streamlit  # Auto-start on reboot
sudo systemctl status streamlit  # Check status
```

**Option B: supervisor**
```bash
sudo apt-get install supervisor

# Create config
sudo nano /etc/supervisor/conf.d/streamlit.conf
```

Paste:
```ini
[program:streamlit]
directory=/path/to/hindi-asr-tts
command=/path/to/venv/bin/streamlit run app.py --server.port 8501
autostart=true
autorestart=true
stderr_logfile=/var/log/streamlit.err.log
stdout_logfile=/var/log/streamlit.out.log
```

#### Step 6: Add Reverse Proxy (nginx)
```bash
sudo apt-get install nginx

sudo nano /etc/nginx/sites-available/default
```

Paste:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

Restart nginx:
```bash
sudo systemctl restart nginx
```

#### Step 7: Enable HTTPS (Free SSL)
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Comparison Table

| Feature | Streamlit Cloud | Docker | Self-Hosted |
|---------|-----------------|--------|-------------|
| **Setup Time** | 5 min | 10 min | 30 min |
| **Cost** | Free | $5-20/mo | $5-100/mo |
| **Maintenance** | None | Minimal | High |
| **Scaling** | Automatic | Manual | Manual |
| **Custom Domain** | ✅ Paid | ✅ | ✅ |
| **Uptime SLA** | ~99% | Your choice | Your choice |
| **Best For** | Most users | Production | Advanced |

---

## Troubleshooting Deployment

### Issue: "ModuleNotFoundError"
**Solution**: Update `requirements.txt`
```bash
pip install -r requirements.txt
```

### Issue: "Out of memory"
**Solution**: 
- Use float16: Already enabled in your code ✅
- Increase server RAM
- Use Streamlit Cloud (scales automatically)

### Issue: "API key not found"
**Solution**:
- Streamlit Cloud: Add in Settings → Secrets
- Docker: Pass environment variables
- Self-hosted: Create .env file

### Issue: "App is slow"
**Solution**:
- First run downloads the model (~1.5GB) - caches after
- Use Streamlit Cloud for auto-scaling
- Upgrade server RAM/CPU

### Issue: "Port already in use"
**Solution**:
```bash
# Change port
streamlit run app.py --server.port 8502
```

---

## Post-Deployment

### 1. Monitor Your App
- **Streamlit Cloud**: Dashboard shows app health
- **Docker**: Use `docker logs` or monitoring tools
- **Self-hosted**: Check systemd logs

### 2. Monitor API Usage
- **HF_TOKEN**: https://huggingface.co/settings/keys
- **SARVAM_API_KEY**: https://console.sarvam.ai/

### 3. Update Your App
- **Streamlit Cloud**: Auto-updates on git push ✅
- **Docker**: Rebuild and push new image
- **Self-hosted**: Git pull and restart service

### 4. Share Your Deployment
- Streamlit Cloud: Share URL directly
- Docker/Self-hosted: Use your domain
- Add to README.md

---

## Production Best Practices

### 1. Rate Limiting
Add to your app (protects API):
```python
from datetime import datetime, timedelta

# Track requests per user
request_count = {}
REQUEST_LIMIT = 10  # per hour
TIME_WINDOW = 3600  # seconds

def check_rate_limit(user_id):
    now = datetime.now()
    if user_id not in request_count:
        request_count[user_id] = []
    
    # Remove old requests
    request_count[user_id] = [
        t for t in request_count[user_id] 
        if (now - t).total_seconds() < TIME_WINDOW
    ]
    
    if len(request_count[user_id]) >= REQUEST_LIMIT:
        return False
    
    request_count[user_id].append(now)
    return True
```

### 2. Error Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # your code
except Exception as exc:
    logger.error(f"Processing failed: {exc}")
```

### 3. Health Check Endpoint
For Docker:
```python
@st.cache_data
def health_check():
    return {"status": "healthy", "model": "loaded"}
```

### 4. Backup & Monitoring
- Streamlit Cloud: Automatic ✅
- Docker: Use container orchestration (Kubernetes)
- Self-hosted: Set up monitoring (Prometheus, Grafana)

---

## My Recommendation

### For You (Right Now):
**→ Use Streamlit Cloud** (5 minutes to live!)

1. Push to GitHub
2. Sign up at streamlit.io/cloud
3. Click "New app"
4. Add secrets
5. Done! ✅

### Why Streamlit Cloud?
- ✅ Free tier available
- ✅ Auto-scales
- ✅ HTTPS/SSL included
- ✅ No DevOps needed
- ✅ Perfect for research/demos

---

## Quick Start Commands

### Streamlit Cloud
```bash
git push origin main
# Then go to https://streamlit.io/cloud
```

### Docker
```bash
docker build -t hindi-asr-tts .
docker run -e HF_TOKEN="..." -e SARVAM_API_KEY="..." -p 8501:8501 hindi-asr-tts
```

### Self-Hosted (Linux)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Support

- **Streamlit Cloud Issues**: https://discuss.streamlit.io
- **Docker Issues**: https://forums.docker.com
- **Your App Issues**: Check app.py error messages

---

**Recommended**: Start with **Streamlit Cloud** for fastest deployment! 🚀
