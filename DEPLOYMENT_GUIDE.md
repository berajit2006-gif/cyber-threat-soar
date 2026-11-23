# 🚀 Deployment Guide: Autonomous AI SOC & XAI Platform

Access your SOC Command Center from anywhere in the world using one of the methods below.

---

## ⚡ Method 1: Instant Public Link (30 Seconds — Zero Setup)

Use this method to immediately test or demonstrate the app from your phone, to professors, or teammates while running on your machine.

### Option A: Using Cloudflare Tunnel (Free, No Signup Required)
Run in PowerShell:
```powershell
winget install --id Cloudflare.cloudflared -e
cloudflared tunnel --url http://127.0.0.1:8000
```
It will output a live HTTPS URL such as:
`https://random-words.trycloudflare.com` -> Accessible worldwide!

### Option B: Using ngrok
```powershell
ngrok http 8000
```
Copy the Forwarding HTTPS address (e.g., `https://xxxx.ngrok-free.app`).

---

## 🌐 Method 2: Free 24/7 Cloud Hosting (Render.com)

Render provides free hosting with automatic HTTPS certificates and continuous deployment from GitHub.

1. **Initialize Git & Push to GitHub**:
   ```bash
   cd "cyber-threat-soar"
   git init
   git add .
   git commit -m "Initial SOC Platform release"
   # Push to your GitHub account repository
   git remote add origin https://github.com/<your-username>/cyber-threat-soar.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com) and create a free account.
   - Click **New +** -> **Web Service**.
   - Connect your GitHub repository.
   - Set the settings:
     - **Name**: `cyber-soc-platform`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt && python train_pipeline.py`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Instance Type**: `Free`
   - Click **Create Web Service**.
   - Your permanent URL will be: `https://cyber-soc-platform.onrender.com`.

---

## 🐳 Method 3: Containerized Cloud Deploy (Railway / Fly.io / HuggingFace)

A production-grade `Dockerfile` and `docker-compose.yml` are already created in the project root:

### Option A: Deploy on Railway (railway.app)
1. Go to [railway.app](https://railway.app).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select this repository. Railway automatically detects `Dockerfile` and deploys it.
4. Go to **Settings** -> **Generate Domain** to get a public URL.

### Option B: Free Docker Space on Hugging Face Spaces
1. Create a Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Space SDK: **Docker** (Blank).
3. Clone the repo and push the `cyber-threat-soar` contents.
4. Hugging Face builds and hosts the app on their high-performance cloud for free.

---

## 🖥️ Method 4: Self-Hosted Cloud VPS (AWS EC2 / DigitalOcean / GCP)

To run on a dedicated Linux cloud server (Ubuntu 22.04 / 24.04):

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/cyber-threat-soar.git
cd cyber-threat-soar

# 2. Run with Docker Compose
docker compose up -d --build

# 3. Allow inbound firewall traffic on port 8000
sudo ufw allow 8000/tcp
```
Access at `http://<YOUR_SERVER_PUBLIC_IP>:8000`.
