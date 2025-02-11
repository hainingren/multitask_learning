if __name__ == "__main__":
    # import sys
    # import os
    
    # sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Ensure project root is in path
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))  # Add src/
    # print("PYTHONPATH:", sys.path) 
    from src.sentence_transformer import SentenceTransformer
    from src.multitask_model import MultiTaskSentenceTransformer

    import torch
    model = SentenceTransformer(model_name="distilbert-base-uncased", pool_mode="cls")

    sample_sentences = [
        "Hello world.",
        "Twinkle Twinkle little star, how I wish what you are!",
        "Have you ever heard Mozart's violin concertos?"
    ]

    with torch.no_grad():
        embeddings = model.forward(sample_sentences)  # shape: (3, hidden_size)
    
    print("Embeddings shape:", embeddings.shape)
    print("Sample embedding for first sentence:\n", embeddings[0])

    multi_task_model = MultiTaskSentenceTransformer(
        backbone_model=model,
        num_classes_taskA=10,  # e.g., 3 classes for sentence classification
        num_classes_taskB=20   # e.g., 2 classes for sentiment
    )

    outputs = multi_task_model.forward(sample_sentences)
     
    print("Task A logits:", outputs["taskA"])  # shape: (2, 3)
    print("Task B logits:", outputs["taskB"])  # shape: (2, 2)