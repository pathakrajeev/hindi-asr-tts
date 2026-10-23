# Pre-Upload Checklist ✅

Before pushing to GitHub, verify everything below:

## 🔐 Secrets Safety

- [ ] `.env` is in `.gitignore` ✅
- [ ] `.streamlit/secrets.toml` is in `.gitignore` ✅
- [ ] `.env.example` has PLACEHOLDER values only (no real keys)
- [ ] No API keys appear in `app.py`, `README.md`, or code comments
- [ ] Ran `git status` and confirmed `.env` is not staged

**Verify with:**
```bash
cat .gitignore                           # Confirm .env is listed
git status                               # Verify .env is not staged
grep -r "hf_" --include="*.py"          # Search for real tokens
grep -r "sk_" --include="*.py"          # Search for real API keys
```

## 📝 Documentation

- [ ] `README.md` has setup instructions
- [ ] `SECURITY.md` explains secrets handling
- [ ] `.env.example` shows all required keys
- [ ] No hardcoded paths or credentials in docs

## 💾 Code Quality

- [ ] No debug print statements left
- [ ] No commented-out code
- [ ] Error handling is secure (no API responses leaked)
- [ ] All imports are used
- [ ] Code follows PEP 8 style

**Quick check:**
```bash
# Check for debug prints
grep -n "print(" app.py | grep -v "st\."

# Check for commented code
grep -n "^#.*=" app.py | head -5
```

## 🧪 Testing

- [ ] App runs locally: `streamlit run app.py`
- [ ] Missing API keys show helpful error messages (not crashes)
- [ ] File uploads validate correctly
- [ ] Error messages are user-friendly

## 📦 Project Files

- [ ] `requirements.txt` is up-to-date
- [ ] `.gitignore` excludes: `.env`, `.streamlit/secrets.toml`, `__pycache__/`, `.venv/`
- [ ] No large files committed (check: `git ls-files | xargs wc -l | sort -n | tail`)

## 🚀 Git Setup

```bash
# Initialize repo (if not done)
git init
git add .
git commit -m "Initial commit: Hindi Live ASR → TTS Streamlit app"

# Check what you're about to push
git log --oneline -5
git status

# BEFORE pushing, verify:
git diff HEAD .env          # Should error: .env is in .gitignore
```

## 📋 Final Verification

**Run these commands:**

```bash
# 1. Check for secrets
grep -rE "hf_|sk_|token|secret" . \
  --include="*.py" \
  --include="*.md" \
  --include="*.txt" \
  --include="*.yml" \
  --include="*.yaml" \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude=.env \
  --exclude=secrets.toml

# 2. Verify .gitignore is working
git check-ignore .env
git check-ignore .streamlit/secrets.toml

# 3. List files to be committed
git status
```

**If all passes:** ✅ Ready to push!

```bash
git remote add origin https://github.com/yourusername/hindi-live-streamlit.git
git branch -M main
git push -u origin main
```

---

## ⚠️ If you accidentally committed secrets:

**DO NOT** just remove and push. The history still contains them:

```bash
# Remove from git history (ONLY if not yet public)
git rm --cached .env
git commit --amend --no-edit
git push -f

# Better: Use BFG Repo-Cleaner
bfg --delete-files .env
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push -f
```

**Then:**
1. Rotate your API keys immediately
2. Create new HF token and Sarvam API key
3. Update `.env` with new credentials

---

## 📚 Next Steps

After pushing to GitHub:

1. **Set up Streamlit Cloud** (optional but recommended)
   - Go to https://streamlit.io/cloud
   - Connect your repo
   - Add secrets in Settings → Secrets
   - App will auto-deploy

2. **Monitor your API keys**
   - Check Hugging Face usage: https://huggingface.co/settings/keys
   - Check Sarvam usage: https://console.sarvam.ai/
   - Set rate limits if available

3. **README visibility**
   - Ensure users see `.env.example` and SECURITY.md
   - Keep setup instructions up-to-date

---

Good luck! 🚀
