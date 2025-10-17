import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import optuna
from torch import nn
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import EarlyStoppingCallback
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multi-GPU

    # For deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    optuna.seed = seed

# Preprocess Dataset
splits = {'train': 'data/1123/toxic-chat_annotation_train.csv', 'test': 'data/1123/toxic-chat_annotation_test.csv'}
train_df = pd.read_csv("hf://datasets/lmsys/toxic-chat/" + splits["train"])
test_df = pd.read_csv("hf://datasets/lmsys/toxic-chat/" + splits["test"])
df = pd.concat([train_df, test_df]).reset_index(drop=True)

train_df = df.sample(frac=0.8, random_state=42)
validation_df = df.drop(train_df.index).sample(frac=0.5, random_state=42)
test_df = df.drop(validation_df.index).drop(train_df.index)

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Import Pre-trained BERT
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Define dataset
max_len = 128
train_dataset = ClassificationDataset(train_df['user_input'], train_df['toxicity'], tokenizer, max_len)
val_dataset = ClassificationDataset(validation_df['user_input'], validation_df['toxicity'], tokenizer, max_len)
test_dataset = ClassificationDataset(test_df['user_input'], test_df['toxicity'], tokenizer, max_len)

# Balance classes
labels = train_df["toxicity"].values
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.01
)

def model_init_lora():
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)
    lora_config = LoraConfig(
        r=16,  # Best trial parameter
        lora_alpha=32, # Best trial parameter
        target_modules=["query", "value"],
        lora_dropout=0.1, # Best trial parameter
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    return model

def train_lora_model():
    print("Training LoRA model with best parameters...")
    lora_model = model_init_lora()
    training_args = TrainingArguments(
        output_dir="./results_lora",
        learning_rate=3e-05, # Best trial parameter
        per_device_train_batch_size=16, # Best trial parameter
        per_device_eval_batch_size=16, # Best trial parameter
        num_train_epochs=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0, # Best trial parameter
        logging_dir="./logs_lora",
        report_to=["wandb"],
        logging_steps=50,
        save_total_limit=2,
        seed=42
    )

    trainer = WeightedTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        class_weights=class_weights
    )

    trainer.train()
    print("LoRA model training finished.")
    return trainer

def model_init_full():
    return AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)

def train_full_model():
    print("Training full fine-tuned model with best parameters...")
    full_model = model_init_full()
    full_training_args = TrainingArguments(
        output_dir="./results_full",
        learning_rate=1e-05, # Best trial parameter
        per_device_train_batch_size=32, # Best trial parameter
        per_device_eval_batch_size=32, # Best trial parameter
        num_train_epochs=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0.0, # Best trial parameter
        logging_dir="./logs_full",
        report_to=["wandb"],
        logging_steps=50,
        save_total_limit=2,
        seed=42
    )
    full_trainer = WeightedTrainer(
        model=full_model,
        args=full_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        class_weights=class_weights
    )

    full_trainer.train()
    print("Full fine-tuned model training finished.")
    return full_trainer

if __name__ == "__main__":
    set_seed(42)

    # Train and evaluate LoRA model
    lora_trainer = train_lora_model()
    lora_metrics = lora_trainer.evaluate(test_dataset)
    print("\nLoRA Model Performance on Test Set After Training:")
    print(lora_metrics)

    # Train and evaluate full fine-tuned model
    full_trainer = train_full_model()
    full_metrics = full_trainer.evaluate(test_dataset)
    print("\nFull Fine-tuned Model Performance on Test Set After Training:")
    print(full_metrics)