from datasets import load_dataset
from src.config import Config

def load_and_preprocess():
    """
    Load XSum dataset and preprocess
    """
    dataset = load_dataset("xsum", split="train[:5000]")  
    
    dataset = dataset.map(lambda x: {
        "machine_text": x["document"] 
    })
    
    return dataset
