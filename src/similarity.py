from bart_score import BARTScorer
from src.config import Config

class SimilarityCalculator:
    def __init__(self):
        self.scorer = BARTScorer(
            device="cuda" if torch.cuda.is_available() else "cpu",
            checkpoint=Config.BART_CHECKPOINT
        )
    
    def compute_similarity(self, original: str, proofread: str) -> float:
        return self.scorer.score([original], [proofread])[0]
