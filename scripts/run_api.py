from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "rlsat.api:app",
        host=os.environ.get("RLSAT_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("RLSAT_API_PORT", "8000")),
        reload=os.environ.get("RLSAT_API_RELOAD", "0") == "1",
    )
