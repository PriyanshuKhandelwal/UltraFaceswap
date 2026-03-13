#!/usr/bin/env python3
"""Run UltraFaceswap API server."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load .env if present (for ULTRAFACESWAP_FACEFUSION_PATH, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    reload = os.environ.get("ULTRAFACESWAP_RELOAD", "false").lower() == "true"
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=reload,
    )
