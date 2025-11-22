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
    
    # Freeze all model parameters (we don't want to update the model)
    for param in model.parameters():
        param.requires_grad = False
    
    # But we DO need gradients w.r.t. embeddings for GCG attack
    # The gradient will be computed on the one-hot input, not the embedding weights
    # So we need to ensure the embedding layer can participate in the computational graph
    embed_layer = model.get_input_embeddings()
    for param in embed_layer.parameters():
        param.requires_grad = True
        
    return model, tokenizer
