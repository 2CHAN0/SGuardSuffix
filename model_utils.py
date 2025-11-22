import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import Config

def load_model_and_tokenizer():
    """
    Loads the SGuard model and tokenizer.
    """
    print(f"Loading model: {Config.MODEL_ID} on {Config.DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=torch.float16 if Config.DEVICE != "cpu" else torch.float32,
        device_map=Config.DEVICE,
        trust_remote_code=True
    )
    
    model.eval()
    
    # Ensure we can compute gradients w.r.t inputs (embeddings)
    # This is crucial for GCG attack
    for param in model.parameters():
        param.requires_grad = False
        
    return model, tokenizer
