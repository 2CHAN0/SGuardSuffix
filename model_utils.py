import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import Config

def load_model_and_tokenizer(model_name_or_path=None):
    """
    Loads the SGuard model and tokenizer.
    """
    if model_name_or_path is None:
        model_name_or_path = Config.MODEL_ID
        
    print(f"Loading model: {model_name_or_path} on {Config.DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
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
