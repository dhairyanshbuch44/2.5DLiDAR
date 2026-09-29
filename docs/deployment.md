# Fovea-LiDAR Cloud Deployment Guide

This guide explains how to deploy Fovea-LiDAR to the cloud so that it runs **24/7 forever with persistent real-time WebSockets**.

---

## 1. Why Vercel Alone Cannot Run This "Forever"

Fovea-LiDAR is **not** a traditional static website or a short-lived REST API:
- It runs a **continuous real-time Python perception loop** (`10 Hz - 20 Hz`) calculating 3D Kalman filters, dynamic foveation, and multi-resolution grids.
- It streams frames to the browser over a **persistent, long-lived WebSocket connection (`/ws/stream`)**.
- It requires scientific Python packages (`numpy`, `scipy`) that exceed serverless bundle constraints.

**Vercel's Architecture**:
- Vercel functions are **serverless (AWS Lambda)** with execution timeouts (10s to 60s max).
- Vercel **kills persistent WebSocket daemon loops** and puts idle functions to sleep.
- Therefore, Vercel **cannot** host the continuous Python WebSocket engine by itself.

---

## 2. Recommended Solution: Hugging Face Spaces (100% Free & Always-On)

**Hugging Face Spaces** is the gold standard for AI and robotics projects:
- **100% Free**
- **2 vCPUs + 16 GB RAM** (perfect for 3D LiDAR point clouds)
- **Persistent Docker Container** (runs 24/7, never times out)
- **Native WebSockets (`wss://`) supported out of the box**

### Steps to Deploy on Hugging Face (Takes 3 Minutes):
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set:
   - **Space name**: `fovea-lidar`
   - **License**: `MIT` or `Apache 2.0`
   - **Space SDK**: Select **`Docker`** $\rightarrow$ **`Blank`**
   - **Space hardware**: `Free (2 vCPU, 16GB RAM)`
3. Push or upload your repository files (including the newly created [`Dockerfile`](../Dockerfile)).
4. Hugging Face will automatically build your Docker container and launch the app at:
   `https://<your-username>-fovea-lidar.hf.space`

---

## 3. Alternative Solution: Render.com (Free Web Service)

Render provides free persistent Docker container hosting with full WebSocket support:
1. Create a free account at [render.com](https://render.com).
2. Click **"New +"** $\rightarrow$ **"Web Service"**.
3. Connect your GitHub/GitLab repository.
4. Render will automatically detect [`render.yaml`](../render.yaml) or your [`Dockerfile`](../Dockerfile).
5. Select the **Free** tier and click **"Create Web Service"**.
6. Your live dashboard will be accessible at:
   `https://fovea-lidar.onrender.com`

---

## 4. Hybrid Solution: Vercel Frontend + Cloud Backend

If you specifically want a **`.vercel.app`** domain for your frontend:
1. Deploy the backend WebSocket server to **Hugging Face** or **Render** (as shown above) to get your live WebSocket URL (e.g. `fovea-lidar.onrender.com`).
2. Deploy the repository to **Vercel** via GitHub integration (it uses [`vercel.json`](../vercel.json) to serve the static 3D HTML5/WebGL dashboard).
3. Access your Vercel URL with the `backend` parameter:
   ```text
   https://fovea-lidar.vercel.app/?backend=fovea-lidar.onrender.com
   ```
   The frontend will automatically establish the secure WebSocket connection (`wss://`) to your live backend.
