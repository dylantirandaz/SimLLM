import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    #OpenAI
    OPENAI_API_KEY = os.getenv(OPENAI_KEY) #stored in secret manager
    OPENAI_MODEL = "gpt-3.5-turbo"

    #PATHS
    DATA_DIR = "data/xsum"
    MODEL_SAVE_DIR = "models/"

    #BERT/RoBERTa
    BART_CHECKPOINT = "facebook/bart-large-cnn"
    ROBERTA_CHECKPOINT = "roberta-base"

    #TRAINING
    BATCH_SIZE = 64
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
