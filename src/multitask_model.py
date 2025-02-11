# multi_task_model.py

import torch
import torch.nn as nn

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, 
                 backbone_model: nn.Module,
                 num_classes_taskA: int = 3,
                 num_classes_taskB: int = 3):
        """
        :param backbone_model: A SentenceTransformer model that outputs embeddings
        :param num_classes_taskA: Number of classes for Task A (sentence classification)
        :param num_classes_taskB: Number of classes for Task B (e.g., sentiment)
        """
        super().__init__()
        self.backbone = backbone_model

        # Task-specific heads
        # Task A: Sentence Classification
        self.classifier_taskA = nn.Sequential(
            nn.Linear(self.backbone.model.config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, num_classes_taskA)
        )

        # Task B: Sentiment Analysis (or any other classification)
        self.classifier_taskB = nn.Sequential(
            nn.Linear(self.backbone.model.config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, num_classes_taskB)
        )

    def forward(self, sentences):
        """
        :param sentences: List of strings (or tokenized batch).
        :return: Dictionary with 'taskA' and 'taskB' logits
        """
        # 1. Get sentence embeddings from the shared backbone
        embeddings = self.backbone.forward(sentences)  # shape: (batch_size, hidden_size)

        # 2. Forward pass through each task head
        logits_taskA = self.classifier_taskA(embeddings)
        logits_taskB = self.classifier_taskB(embeddings)

        return {
            "taskA": logits_taskA,  # shape: (batch_size, num_classes_taskA)
            "taskB": logits_taskB   # shape: (batch_size, num_classes_taskB)
        }
