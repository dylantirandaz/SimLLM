# run.py
import argparse
from src.proofread import proofread_text
from src.similarity import SimilarityCalculator
from src.dataset import load_and_preprocess
from src.train import train
from src.predict import SimLLMDetector

def generate_data(sample_text: str):
    """Step 1: Proofread text and calculate similarity"""
    print("\n=== Generating Data ===")
    
    proofread = proofread_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Proofread: {proofread}")

    similarity = SimilarityCalculator().compute_similarity(sample_text, proofread)
    print(f"BART Similarity Score: {similarity:.2f}")
    
    return similarity

def train_model():
    """Step 2: Train classifier"""
    print("\n=== Training Model ===")
    train()
    print("Training completed! Model saved to models/")

def detect_text():
    """Step 3: Detect machine-generated text"""
    print("\n=== Running Detection ===")
    detector = SimLLMDetector()

  #sample
    human_text = "The quick brown fox jumps over the lazy dog."
    machine_text = "The rapid auburn vulpine leaps across the indolent canine."
    
    print(f"Human text score: {detector.predict(human_text):.2f}")
    print(f"Machine text score: {detector.predict(machine_text):.2f}")

def main():
    parser = argparse.ArgumentParser(description="Run SimLLM pipeline")
    parser.add_argument('--train', action='store_true', help="Run training only")
    parser.add_argument('--detect', action='store_true', help="Run detection only")
    
    args = parser.parse_args()

    sample_text = "Large language models are transforming AI applications."
    
    try:
        if not args.detect:
            if not args.train:
                generate_data(sample_text)
            train_model()
        
        if not args.train:
            detect_text()
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
