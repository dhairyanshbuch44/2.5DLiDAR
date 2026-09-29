"""Cloud deployment entry point for Fovea-LiDAR.

Supports:
1. Native Python cloud runtimes (Render, Railway, Koyeb)
2. Hugging Face Spaces (Gradio SDK - 100% Free with No Docker Required)
"""

import os
import sys
from pathlib import Path

# Ensure repo root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import uvicorn
from dashboard.server import app

# For Hugging Face Spaces with Gradio SDK (100% free, no credit card or Docker needed)
try:
    import gradio as gr
    demo = gr.mount_gradio_app(app, gr.Blocks(), path="/")
except Exception:
    demo = None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=port)
