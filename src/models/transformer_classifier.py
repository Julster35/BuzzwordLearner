"""
Transformer-based classifier for job titles with support for class imbalance.

Includes WeightedTrainer for handling imbalanced datasets.
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


class JobTitleDataset(Dataset):
    """Simple dataset for job title classification."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    """
    Trainer with class-weighted loss for imbalanced datasets.
    
    Computes CrossEntropyLoss with class weights that penalize
    errors on minority classes more heavily.
    """
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = torch.tensor(
                self.class_weights, 
                device=logits.device, 
                dtype=logits.dtype
            )
            loss_fn = CrossEntropyLoss(weight=weight)
        else:
            loss_fn = CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class TransformerClassifier:
    """
    Transformer-based classifier for job titles.

    Supports two-stage training:
    1. Pre-train on CSV patterns (high volume, curated)
    2. Fine-tune on LinkedIn CV data (domain adaptation)
    
    Features:
    - Optional class-weighted loss for imbalanced datasets
    - Early stopping with validation set
    - GPU acceleration when available
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        num_labels: int = 2,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the classifier.

        Args:
            model_name: Hugging Face model name or path to local model
            num_labels: Number of classification labels
            id2label: Mapping from label ID to label name
            label2id: Mapping from label name to label ID
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = id2label or {}
        self.label2id = label2id or {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def train(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        output_dir: str = "./results",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        warmup_ratio: float = 0.1,
        use_class_weights: bool = False
    ):
        """
        Fine-tune the model on the provided texts and labels.
        
        Args:
            texts: Training texts
            labels: Training labels (as integers)
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            output_dir: Directory to save checkpoints
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_ratio: Proportion of training for learning rate warmup
            use_class_weights: Whether to use balanced class weights (for imbalanced data)
        """
        print(f"Training on {len(texts)} examples...")

        # Tokenize
        train_encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=128
        )
        train_dataset = JobTitleDataset(train_encodings, labels)

        # Validation set
        if val_texts is not None and val_labels is not None:
            val_encodings = self.tokenizer(
                val_texts, truncation=True, padding=True, max_length=128
            )
            val_dataset = JobTitleDataset(val_encodings, val_labels)
            eval_strategy = "epoch"
        else:
            val_dataset = None
            eval_strategy = "no"
        
        # Calculate class weights if requested
        class_weights = None
        if use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            unique_labels = np.unique(labels)
            class_weights = compute_class_weight(
                'balanced', 
                classes=unique_labels, 
                y=labels
            )
            print(f"Using class weights: {dict(zip(unique_labels, class_weights))}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=warmup_ratio,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy=eval_strategy,
            save_strategy=eval_strategy,
            load_best_model_at_end=val_dataset is not None,
            learning_rate=learning_rate,
            report_to="none",
            save_total_limit=2,
        )

        # Callbacks
        callbacks = []
        if val_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

        # Use WeightedTrainer if class weights are specified
        if class_weights is not None:
            trainer = WeightedTrainer(
                class_weights=class_weights,
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=callbacks
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=callbacks
            )

        trainer.train()

        # Keep the best model
        if val_dataset:
            self.model = trainer.model

        print("Training complete!")

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        """
        Predict label IDs for a list of texts.

        Args:
            texts: Input texts
            batch_size: Batch size for inference

        Returns:
            List of predicted label IDs
        """
        self.model.eval()
        all_predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().tolist())

        return all_predictions

    def predict_labels(self, texts: List[str], batch_size: int = 32) -> List[str]:
        """
        Predict label names for a list of texts.

        Args:
            texts: Input texts
            batch_size: Batch size for inference

        Returns:
            List of predicted label names
        """
        label_ids = self.predict(texts, batch_size)
        return [self.id2label.get(lid, str(lid)) for lid in label_ids]

    def predict_with_confidence(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """
        Predict labels with confidence scores.

        Args:
            texts: Input texts
            batch_size: Batch size for inference

        Returns:
            List of (label_name, confidence) tuples
        """
        self.model.eval()
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                confidences, predictions = torch.max(probs, dim=-1)

                for pred, conf in zip(predictions.cpu().tolist(), confidences.cpu().tolist()):
                    label = self.id2label.get(pred, str(pred))
                    results.append((label, conf))

        return results

    def save(self, path: Union[str, Path]):
        """Save the model, tokenizer, and custom config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save custom config
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "id2label": self.id2label,
            "label2id": self.label2id
        }
        with open(path / "classifier_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TransformerClassifier":
        """
        Load a saved model.

        Args:
            path: Path to the saved model directory

        Returns:
            Loaded TransformerClassifier
        """
        path = Path(path)

        with open(path / "classifier_config.json", "r") as f:
            config = json.load(f)

        # Create instance loading from local path
        instance = cls(
            model_name=str(path),
            num_labels=config["num_labels"],
            id2label={int(k): v for k, v in config["id2label"].items()},
            label2id=config["label2id"]
        )

        return instance
