#!/bin/bash
# Deploy UltraFaceswap to Hugging Face Spaces
#
# Usage:
#   1. Create a Space at huggingface.co (Docker SDK, T4 GPU)
#   2. Set your HF username and space name below
#   3. Run: bash scripts/deploy_hf.sh
#
# Prerequisites:
#   - huggingface-cli installed: pip install huggingface_hub
#   - Logged in: huggingface-cli login

set -e

HF_USERNAME="${HF_USERNAME:-YOUR_USERNAME}"
SPACE_NAME="${SPACE_NAME:-ultrafaceswap}"
SPACE_REPO="${HF_USERNAME}/${SPACE_NAME}"

echo "=== UltraFaceswap → Hugging Face Spaces ==="
echo "Space: https://huggingface.co/spaces/${SPACE_REPO}"
echo ""

# 1. Build frontend
echo "[1/5] Building frontend..."
cd "$(dirname "$0")/.."
(cd frontend && npm install && npm run build)

# 2. Prepare deployment directory
echo "[2/5] Preparing deployment..."
DEPLOY_DIR=$(mktemp -d)
trap "rm -rf $DEPLOY_DIR" EXIT

# Copy only what's needed
cp Dockerfile.hf "$DEPLOY_DIR/Dockerfile"
cp README.hf.md "$DEPLOY_DIR/README.md"
cp requirements.txt "$DEPLOY_DIR/"
cp run_api.py "$DEPLOY_DIR/"
cp -r backend "$DEPLOY_DIR/"
cp -r assets "$DEPLOY_DIR/"
cp -r frontend/dist "$DEPLOY_DIR/frontend/dist" 2>/dev/null || mkdir -p "$DEPLOY_DIR/frontend/dist"

# Remove __pycache__
find "$DEPLOY_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# 3. Init git in deploy dir
echo "[3/5] Initializing git..."
cd "$DEPLOY_DIR"
git init
git lfs install
git add .
git commit -m "Deploy UltraFaceswap"

# 4. Push to HF
echo "[4/5] Pushing to Hugging Face..."
git remote add hf "https://huggingface.co/spaces/${SPACE_REPO}"
git push hf main --force

echo ""
echo "[5/5] Done! Your Space is building at:"
echo "  https://huggingface.co/spaces/${SPACE_REPO}"
echo ""
echo "Don't forget to set secrets in Space settings:"
echo "  - TELEGRAM_BOT_TOKEN  (from @BotFather)"
echo "  - TELEGRAM_WEBHOOK_URL (https://${HF_USERNAME}-${SPACE_NAME}.hf.space)"
