"""Quick test: can we reach MiroFish and get a prediction?"""

import os
import sys
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def test_health(base_url: str) -> bool:
    """Check if MiroFish is running."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        data = r.json()
        print(f"  Health: {data}")
        return data.get("status") == "ok"
    except Exception as e:
        print(f"  Health check failed: {e}")
        return False


def test_predict_adapter(base_url: str) -> bool:
    """Test our PredictAdapter against live MiroFish."""
    from adapters.predict import PredictAdapter

    adapter = PredictAdapter(base_url=base_url)

    if not adapter.is_mirofish_available():
        print("  MiroFish not available — adapter will use Claude fallback")
        return False

    print("  MiroFish is available! Testing prediction...")
    result = adapter.run_prediction(
        "Will the Federal Reserve cut interest rates in June 2026?",
        market_data={"yes_price": 0.42, "volume": 500000, "end_date": "2026-06-30"},
    )

    print(f"  Probability: {result.probability:.0%}")
    print(f"  Confidence:  {result.confidence}")
    print(f"  Reasoning:   {result.reasoning[:200]}")
    return True


def main():
    base_url = os.environ.get("MIROFISH_BASE_URL", "http://127.0.0.1:5001")
    print(f"Testing MiroFish at {base_url}\n")

    print("[1] Health check...")
    healthy = test_health(base_url)

    if not healthy:
        print("\nMiroFish is not running. Start it with:")
        print("  python scripts/start_mirofish.py")
        print("\nOr via Docker:")
        print("  docker compose up mirofish")
        sys.exit(1)

    print("\n[2] Prediction test...")
    ok = test_predict_adapter(base_url)

    if ok:
        print("\n All good! MiroFish is connected and predicting.")
    else:
        print("\n MiroFish is running but prediction failed — check API keys.")


if __name__ == "__main__":
    main()
