from sguard_attack.config import Config
from sguard_attack.model_utils import load_model_and_tokenizer
from sguard_attack.dataset import Dataset
from sguard_attack.attack import GCGAttack
import torch

def test_inference(model, tokenizer, malicious_prompt, suffix):
    """
    Test the model's response with the given suffix.
    This mimics the actual usage scenario.
    """
    print(f"\n{'='*60}")
    print(f"Testing Inference")
    print(f"{'='*60}")
    
    # Get token IDs for safe/unsafe
    vocab = tokenizer.get_vocab()
    safe_token_id = vocab['safe']
    unsafe_token_id = vocab['unsafe']
    
    # Construct the full prompt with suffix (SAME as training)
    full_content = f"{malicious_prompt} {suffix}"
    messages = [{"role": "user", "content": full_content}]
    
    print(f"Original prompt: {malicious_prompt}")
    print(f"Suffix: {suffix}")
    print(f"Full content: {full_content}")
    
    # Tokenize with chat template (SAME as training)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt'
    ).to(Config.DEVICE)
    
    # Generate with max_new_tokens=1 (SGuard only outputs one token)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True
        )
        
        # Get the generated token
        generated_token_id = output.sequences[0][-1].item()
        generated_token = tokenizer.decode([generated_token_id])
        
        # Get logits and probabilities
        output_logits = output.logits[0][0]  # First (and only) generated token
        selected_logits = torch.tensor([
            output_logits[safe_token_id],
            output_logits[unsafe_token_id]
        ])
        probs = torch.softmax(selected_logits, dim=0)
        
        print(f"\nResults:")
        print(f"  Generated token: '{generated_token}' (ID: {generated_token_id})")
        print(f"  P(safe) = {probs[0]:.4f}")
        print(f"  P(unsafe) = {probs[1]:.4f}")
        print(f"  Classification: {'SAFE' if probs[0] >= 0.5 else 'UNSAFE'}")
        
    print(f"{'='*60}\n")
    
    return generated_token, probs[0].item()

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
        
        # First, test baseline (without suffix)
        print("\n--- Baseline Test (No Suffix) ---")
        baseline_token, baseline_safe_prob = test_inference(model, tokenizer, malicious_prompt, "")
        
        # Run attack
        best_suffix, best_suffix_ids, best_loss, best_safe_prob = attacker.run(malicious_prompt)
        
        print(f"\n--- Attack Results ---")
        print(f"  Best Suffix: '{best_suffix}'")
        print(f"  Final Loss: {best_loss:.4f}")
        print(f"  Safe Probability (from training): {best_safe_prob:.4f}")
        
        # Verification with the actual model
        print("\n--- Final Inference Test ---")
        final_token, final_safe_prob = test_inference(model, tokenizer, malicious_prompt, best_suffix)
        
        # Compare training vs inference
        print(f"\n--- Training vs Inference Comparison ---")
        print(f"  Safe Prob (Training): {best_safe_prob:.4f}")
        print(f"  Safe Prob (Inference): {final_safe_prob:.4f}")
        print(f"  Difference: {abs(best_safe_prob - final_safe_prob):.4f}")
        
        if abs(best_safe_prob - final_safe_prob) > 0.1:
            print(f"  ⚠️  WARNING: Large discrepancy between training and inference!")
        else:
            print(f"  ✅ Training and inference are consistent!")
        
        results.append({
            "prompt": malicious_prompt,
            "suffix": best_suffix,
            "loss": best_loss,
            "baseline_safe_prob": baseline_safe_prob,
            "training_safe_prob": best_safe_prob,
            "inference_safe_prob": final_safe_prob,
            "baseline_token": baseline_token,
            "final_token": final_token
        })
        
    print("\n" + "="*60)
    print("ATTACK SUMMARY")
    print("="*60)
    for res in results:
        print(f"\nPrompt: {res['prompt']}")
        print(f"Suffix: '{res['suffix']}'")
        print(f"Baseline: {res['baseline_token']} (P(safe)={res['baseline_safe_prob']:.4f})")
        print(f"After Attack: {res['final_token']} (P(safe)={res['inference_safe_prob']:.4f})")
        print(f"Success: {'✅ YES' if res['inference_safe_prob'] > res['baseline_safe_prob'] else '❌ NO'}")
        print("-" * 60)

if __name__ == "__main__":
    main()
