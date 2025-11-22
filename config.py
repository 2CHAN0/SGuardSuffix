import torch

class Config:
    # Model Configuration
    MODEL_ID = "SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    
    # Attack Configuration
    NUM_STEPS = 500
    TOP_K = 256
    BATCH_SIZE = 512 # Batch size for candidate evaluation
    SUFFIX_LENGTH = 15
    
    # Target
    TARGET_STRING = "safe"
    
    # Random Seed
    SEED = 42
