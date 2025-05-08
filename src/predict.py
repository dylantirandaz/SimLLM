from transformers import RobertaForSequenceClassification, RobertaTokenizer
from src.config import Config
import torch

class SimLLMDetector:
    def __init__(self):
        self.model = RobertaForSequenceClassification.from_pretrained(Config.MODEL_SAVE_DIR)
        self.tokenizer = RobertaTokenizer.from_pretrained(Config.ROBERTA_CHECKPOINT)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def predict(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][1].item()  #prob of ai generation intext
