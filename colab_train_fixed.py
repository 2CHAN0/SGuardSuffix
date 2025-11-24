# SGuard Suffix Attack Training (Colab Compatible)
# Copy and paste this code into a Google Colab cell to run the training.

import sys
import os
import datetime
import torch
import json
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# 1. Mount Google Drive
if IN_COLAB:
    drive.mount('/content/drive')

# 2. Setup Environment
# Clone the repository and install dependencies
if IN_COLAB:
    os.system("rm -rf /content/sguard_attack")
    os.system("git clone https://github.com/2CHAN0/SGuardSuffix.git sguard_attack")
    os.system("pip install -r sguard_attack/requirements.txt")
    # Add package to path
    sys.path.append('/content')
else:
    # Local setup assumption
    pass

from sguard_attack.config import Config
from sguard_attack.model_utils import load_model_and_tokenizer
from sguard_attack.dataset import Dataset
from sguard_attack.attack import GCGAttack

# 3. Model Caching (Google Drive)
if IN_COLAB:
    drive_model_dir = "/content/drive/MyDrive/SGuardSuffix/models/SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1"
    
    if os.path.exists(drive_model_dir) and os.listdir(drive_model_dir):
        print(f"Model found in Google Drive: {drive_model_dir}")
        model_path = drive_model_dir
    else:
        print(f"Model not found in Drive. Downloading from Hugging Face: {Config.MODEL_ID}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        print(f"Saving model to Google Drive: {drive_model_dir}")
        tokenizer.save_pretrained(drive_model_dir)
        model.save_pretrained(drive_model_dir)
        model_path = drive_model_dir
        print("Model saved successfully!")
else:
    model_path = None # Use default from config

# 4. Run Training
# Setup Save Directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if IN_COLAB:
    save_base_dir = f"/content/drive/MyDrive/SGuard_Training_Results/{timestamp}"
else:
    save_base_dir = f"./results/{timestamp}"

os.makedirs(save_base_dir, exist_ok=True)
print(f"Saving results to: {save_base_dir}")

# Load Model & Tokenizer
model, tokenizer = load_model_and_tokenizer(model_name_or_path=model_path)

# Load Dataset
dataset = Dataset()

# Initialize Attack
attacker = GCGAttack(model, tokenizer)

print(f"Starting attack on {len(dataset)} prompts...")

results = []

for i, malicious_prompt in enumerate(dataset):
    print(f"\n[{i+1}/{len(dataset)}] Attacking prompt: {malicious_prompt}")
    
    # Run attack
    # Returns: best_suffix, best_suffix_ids, best_loss, best_safe_prob
    best_suffix, best_suffix_ids, best_loss, best_safe_prob = attacker.run(
        malicious_prompt,
        save_dir=save_base_dir,
        save_interval=10
    )
    
    print(f"Result for '{malicious_prompt}':")
    print(f"  Best Suffix: {best_suffix}")
    print(f"  Final Loss: {best_loss:.4f}")
    print(f"  Safe Probability (Training): {best_safe_prob:.4f}")
    
    # Verification with Chat Template
    print("\n--- Inference Test (Chat Template) ---")
    
    vocab = tokenizer.get_vocab()
    # Use encode for consistency with attack.py
    safe_token_id = tokenizer.encode("safe", add_special_tokens=False)[0]
    unsafe_token_id = tokenizer.encode("unsafe", add_special_tokens=False)[0]
    
    full_content = f"{malicious_prompt} {best_suffix}"
    messages = [{"role": "user", "content": full_content}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt'
    ).to(Config.DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True
        )
        
        generated_token_id = output.sequences[0][-1].item()
        generated_token = tokenizer.decode([generated_token_id])
        
        output_logits = output.logits[0][0]
        selected_logits = torch.tensor([
            output_logits[safe_token_id],
            output_logits[unsafe_token_id]
        ])
        probs = torch.softmax(selected_logits, dim=0)
        inference_safe_prob = probs[0].item()
        
    print(f"  Generated token: '{generated_token}' (ID: {generated_token_id})")
    print(f"  P(safe) = {inference_safe_prob:.4f}")
    print(f"  Classification: {'SAFE' if inference_safe_prob >= 0.5 else 'UNSAFE'}")
    
    results.append({
        "prompt": malicious_prompt,
        "suffix": best_suffix,
        "loss": best_loss,
        "training_safe_prob": best_safe_prob,
        "inference_safe_prob": inference_safe_prob,
        "generated_token": generated_token
    })

# Save Final Summary
summary_path = os.path.join(save_base_dir, "final_summary.json")
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nFinal summary saved to {summary_path}")
