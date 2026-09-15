import os
import sys

# Add the project root to sys.path to ensure we can import defenceAgent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

from defenceAgent.retrieval.rag_manager import RAGManager
import logging

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAGManager
    # Assuming the vector DB is located at d:\AgentScope\chroma_db
    rag = RAGManager(persist_directory=r"d:\AgentScope\chroma_db")
    
    # Path to MM-SafetyBench data
    dataset_path = r"d:\AgentScope\defenceAgent\statics\MM-SafetyBench\data"
    checkpoint_path = r"d:\AgentScope\ingestion_checkpoint.json"
    
    print(f"Starting ingestion of MM-SafetyBench from {dataset_path}...")
    try:
        rag.ingest_mmsafetybench(dataset_path, checkpoint_file=checkpoint_path)
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user. Progress has been saved to checkpoint.")
    
    print("\nIngestion complete or interrupted.")
    rag.get_cost_report()
