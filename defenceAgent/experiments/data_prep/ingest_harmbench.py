import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

from defenceAgent.retrieval.rag_manager import RAGManager
import logging

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize RAGManager
    rag = RAGManager(persist_directory=r'd:\AgentScope\chroma_db')
    
    # Path to HarmBench data
    dataset_path = r'd:\AgentScope\defenceAgent\statics\HarmBench'
    
    print(f'Starting ingestion of HarmBench from {dataset_path}...')
    
    # CLEANUP LEGACY DATA
    sources = ['HarmBench/standard', 'HarmBench/contextual', 'HarmBench/copyright']
    for source in sources:
        try:
            logger.info(f'Cleaning up legacy/existing data for {source}...')
            rag.collection.delete(where={'source': source})
        except Exception as e:
            logger.warning(f'Error during cleanup of {source}: {e}')

    try:
        rag.ingest_all_harmbench(dataset_path)
    except KeyboardInterrupt:
        print('\nIngestion interrupted by user.')
    
    print('\nIngestion complete.')
    rag.get_cost_report()
