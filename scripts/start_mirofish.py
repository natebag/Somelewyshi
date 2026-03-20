"""Start MiroFish backend locally for testing."""

import os
import sys

# Add vendor mirofish backend to path
vendor_backend = os.path.join(
    os.path.dirname(__file__), "..", "vendor", "mirofish", "backend"
)
sys.path.insert(0, os.path.abspath(vendor_backend))

# Load our root .env so MiroFish picks up the keys
from dotenv import load_dotenv

root_env = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(root_env, override=True)


def main():
    # Validate keys are present
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY")
    if not os.environ.get("ZEP_API_KEY"):
        missing.append("ZEP_API_KEY")

    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        print("Check your .env file")
        sys.exit(1)

    # Import and run MiroFish
    from app import create_app

    app = create_app()

    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5001"))

    print(f"\nMiroFish backend starting on http://{host}:{port}")
    print(f"  Health check: http://{host}:{port}/health")
    print(f"  API base:     http://{host}:{port}/api/")
    print(f"  LLM model:    {os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')}")
    print()

    app.run(host=host, port=port, debug=True, threaded=True)


if __name__ == "__main__":
    main()
