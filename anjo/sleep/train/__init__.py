"""Training her weights (Task 5): the night's dataset, the LoRA recipe and the GPU runners.

Nothing here imports torch, transformers or peft at module top: the training script
(``colab_train.py``) loads them only inside the GPU runner's environment, and serving never
imports this package (an import-linter contract).
"""
