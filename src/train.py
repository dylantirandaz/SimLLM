import torch
from transformers import (
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaTokenizer
)
from src.config import Config
from src.dataset import load_and_preprocess

tokenizer = RobertaTokenizer.from_pretrained(Config.ROBERTA_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(
        examples["document"],  #human
        examples["machine_text"],  #ai
        padding="max_length",
        truncation=True,
        max_length=512
    )

def train():
    dataset = load_and_preprocess()
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    model = RobertaForSequenceClassification.from_pretrained(
        Config.ROBERTA_CHECKPOINT,
        num_labels=2
    )
    
    training_args = TrainingArguments(
        output_dir=Config.MODEL_SAVE_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    model.save_pretrained(Config.MODEL_SAVE_DIR)
