from sguard_attack.config import Config
from sguard_attack.model_utils import load_model_and_tokenizer
from sguard_attack.dataset import Dataset
from sguard_attack.attack import GCGAttack
import torch

def main():
    # 1. Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. Load Dataset
    dataset = Dataset()
    
    # 3. Initialize Attack
    attacker = GCGAttack(model, tokenizer)
    
    print(f"Starting attack on {len(dataset)} prompts...")
    
    results = []
    
    for i, malicious_prompt in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] Attacking prompt: {malicious_prompt}")
        
        best_suffix, best_suffix_ids, best_loss = attacker.run(malicious_prompt)
        
        print(f"Result for '{malicious_prompt}':")
        print(f"  Best Suffix: {best_suffix}")
        print(f"  Final Loss: {best_loss:.4f}")
        
        # Verification
        full_input = malicious_prompt + " " + best_suffix
        inputs = tokenizer(full_input, return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        print(f"  Model Output (String): {response}")
        
        # Tensor Verification (More Accurate)
        prompt_ids = tokenizer(malicious_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(Config.DEVICE)
        # We need to ensure shapes match for concatenation
        # prompt_ids: [1, P], best_suffix_ids: [S] -> need [1, S]
        if best_suffix_ids.dim() == 1:
            best_suffix_ids = best_suffix_ids.unsqueeze(0)
            
        full_input_ids = torch.cat([prompt_ids, best_suffix_ids], dim=1)
        
        with torch.no_grad():
            outputs_tensor = model.generate(full_input_ids, max_new_tokens=10)
            # Decode only the new tokens
            new_tokens = outputs_tensor[0][full_input_ids.shape[1]:]
            response_tensor = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        print(f"  Model Output (Tensor): {response_tensor}")
        
        results.append({
            "prompt": malicious_prompt,
            "suffix": best_suffix,
            "loss": best_loss,
            "output": response
        })
        
    print("\n--- Attack Summary ---")
    for res in results:
        print(f"Prompt: {res['prompt']}")
        print(f"Suffix: {res['suffix']}")
        print(f"Output: {res['output']}")
        print("-" * 20)

if __name__ == "__main__":
    main()
