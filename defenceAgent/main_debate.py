import argparse
import asyncio
import json
from .orchestrator_debate import DebateOrchestrator
from .models import ModelRegistry

def main() -> None:
    parser = argparse.ArgumentParser(description="DefenceAgent Debate Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to image")
    parser.add_argument("--text", type=str, default=None, help="Text query")
    args = parser.parse_args()

    # Warm up registry
    _ = ModelRegistry()
    
    orchestrator = DebateOrchestrator()
    
    print(f"Running debate for Image: {args.image}, Text: {args.text}")
    
    # Run the async orchestrator
    result = asyncio.run(orchestrator.run(args.image, args.text))
    
    print("\n== Final Verdict ==")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
