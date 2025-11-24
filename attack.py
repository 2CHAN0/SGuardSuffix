import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from .config import Config
import os
import json
import datetime

class GCGAttack:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = Config.DEVICE
        
        # Get safe/unsafe token IDs
        # Use encode to be sure we get the correct ID for the token string
        # We assume "safe" and "unsafe" are single tokens in the vocab
        safe_ids = self.tokenizer.encode("safe", add_special_tokens=False)
        unsafe_ids = self.tokenizer.encode("unsafe", add_special_tokens=False)
        
        if len(safe_ids) != 1 or len(unsafe_ids) != 1:
            print(f"WARNING: 'safe' or 'unsafe' tokenized to multiple tokens: safe={safe_ids}, unsafe={unsafe_ids}")
            # Fallback to first token if multiple
            self.safe_token_id = safe_ids[0]
            self.unsafe_token_id = unsafe_ids[0]
        else:
            self.safe_token_id = safe_ids[0]
            self.unsafe_token_id = unsafe_ids[0]
            
        # Verify token IDs are within embedding range
        vocab_size = self.model.get_input_embeddings().num_embeddings
        print(f"Safe token ID: {self.safe_token_id}, Unsafe token ID: {self.unsafe_token_id}")
        print(f"Model embedding size: {vocab_size}")
        
        if self.safe_token_id >= vocab_size or self.unsafe_token_id >= vocab_size:
            raise ValueError(f"Token IDs out of bounds! safe={self.safe_token_id}, unsafe={self.unsafe_token_id}, vocab_size={vocab_size}")
        
    def get_input_ids_with_suffix(self, malicious_prompt, suffix_str):
        """
        Constructs input IDs using chat template with suffix appended to prompt.
        Returns: (input_ids, suffix_start_idx, suffix_end_idx)
        """
        # Create the full prompt with suffix
        full_content = f"{malicious_prompt} {suffix_str}"
        messages = [{"role": "user", "content": full_content}]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors='pt'
        )[0].to(self.device)
        
        # To find suffix position, tokenize without suffix and compare
        messages_without_suffix = [{"role": "user", "content": malicious_prompt}]
        input_ids_without_suffix = self.tokenizer.apply_chat_template(
            messages_without_suffix,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors='pt'
        )[0].to(self.device)
        
        # Tokenize just the suffix to get its length
        suffix_tokens = self.tokenizer(f" {suffix_str}", add_special_tokens=False, return_tensors='pt').input_ids[0]
        
        # Suffix should be inserted before the generation prompt
        # Find where they differ
        len_without = len(input_ids_without_suffix)
        len_with = len(input_ids)
        suffix_len = len(suffix_tokens)
        
        # Estimate suffix position (this is approximate)
        # Typically, suffix is added near the end, before generation prompt tokens
        suffix_end_idx = len_without
        suffix_start_idx = suffix_end_idx - suffix_len
        
        return input_ids, suffix_start_idx, suffix_end_idx
        
    def token_gradients(self, input_ids, suffix_slice):
        """
        Computes gradients of the loss with respect to the one-hot text input
        at the suffix positions.
        """
        # Create embeddings matrix
        embed_weights = self.model.get_input_embeddings().weight
        vocab_size = embed_weights.shape[0]
        
        # Check for out-of-bounds input_ids
        if torch.any(input_ids >= vocab_size):
            max_id = torch.max(input_ids).item()
            raise ValueError(f"Input IDs contain value {max_id} which is >= vocab size {vocab_size}")
        
        # One-hot encode the input_ids
        one_hot = torch.zeros(
            input_ids.shape[0],
            embed_weights.shape[0],
            device=self.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1, 
            input_ids.unsqueeze(1), 
            torch.ones(one_hot.shape[:2], device=self.device, dtype=embed_weights.dtype)
        )
        one_hot.requires_grad_(True)
        
        # Get embeddings
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        
        # Forward pass
        outputs = self.model(inputs_embeds=input_embeds)
        logits = outputs.logits
        
        # For SGuard: we only care about the FIRST generated token
        # The logit at position -1 predicts the next token
        next_token_logits = logits[0, -1, :]  # [vocab_size]
        
        # Extract logits for safe and unsafe tokens only
        selected_logits = torch.stack([
            next_token_logits[self.safe_token_id],
            next_token_logits[self.unsafe_token_id]
        ])
        
        # Convert to probabilities
        probs = torch.softmax(selected_logits, dim=0)
        
        # Loss: negative log probability of 'safe' token
        # We want to MINIMIZE this (= maximize P(safe))
        loss = -torch.log(probs[0] + 1e-10)
        
        # Backward pass
        loss.backward()
        
        # Get gradients for the suffix tokens
        grad = one_hot.grad.clone()
        grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-10)
        
        return grad[suffix_slice], loss.item(), probs[0].item()

    def sample_control(self, control_toks, grad, batch_size, topk=256):
        """
        Randomly sample candidate tokens based on gradients.
        """
        control_toks = control_toks.to(self.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        # Randomly select a position to modify for each batch item
        # We want to modify one token for each candidate in the batch
        new_token_pos = torch.randint(0, len(control_toks), (batch_size,), device=self.device)
        
        # Get the top-k indices for the suffix tokens: [suffix_len, topk]
        topk_indices = torch.topk(grad, topk, dim=1).indices
        
        # Select the top-k indices for the specific positions we are modifying: [batch_size, topk]
        relevant_topk_indices = topk_indices[new_token_pos]
        
        # Randomly select one of the top-k tokens for each batch item: [batch_size, 1]
        new_token_val = torch.gather(
            relevant_topk_indices,
            1,
            torch.randint(0, topk, (batch_size, 1), device=self.device)
        )
        
        new_control_toks = original_control_toks.scatter_(
            1, 
            new_token_pos.unsqueeze(-1), 
            new_token_val
        )
        
        return new_control_toks

    def evaluate_candidates(self, malicious_prompt, candidates):
        """
        Evaluate a batch of candidate suffixes.
        
        Args:
            malicious_prompt: The malicious prompt string
            candidates: [batch_size, suffix_len] candidate suffix token IDs
        
        Returns:
            losses: [batch_size] loss for each candidate
            safe_probs: [batch_size] probability of 'safe' token for each candidate
        """
        batch_size = candidates.shape[0]
        
        losses = []
        safe_probs = []
        
        # Evaluate each candidate (could be optimized with true batching)
        for i in range(batch_size):
            candidate_suffix_ids = candidates[i]
            candidate_suffix_str = self.tokenizer.decode(candidate_suffix_ids, skip_special_tokens=True)
            
            # Construct input with chat template
            full_content = f"{malicious_prompt} {candidate_suffix_str}"
            messages = [{"role": "user", "content": full_content}]
            
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
                
                # Get logits for the FIRST generated token
                next_token_logits = logits[0, -1, :]
                
                # Extract logits for safe and unsafe tokens
                selected_logits = torch.stack([
                    next_token_logits[self.safe_token_id],
                    next_token_logits[self.unsafe_token_id]
                ])
                
                # Convert to probabilities
                probs = torch.softmax(selected_logits, dim=0)
                
                # Loss and probability
                safe_prob = probs[0].item()
                loss = -torch.log(probs[0] + 1e-10).item()
                
                losses.append(loss)
                safe_probs.append(safe_prob)
        
        return torch.tensor(losses, device=self.device), torch.tensor(safe_probs, device=self.device)

    def run(self, malicious_prompt, save_dir=None, save_interval=10):
        """
        Runs the GCG attack for a single malicious prompt.
        """
        print(f"\n{'='*60}")
        print(f"Starting GCG attack on prompt: '{malicious_prompt}'")
        print(f"{'='*60}\n")
        
        # Initialize suffix
        suffix = "! ! ! ! ! ! ! ! ! !"
        suffix_ids = self.tokenizer(f" {suffix}", return_tensors="pt", add_special_tokens=False).input_ids[0].to(self.device)
        
        best_loss = float('inf')
        best_suffix = suffix
        best_suffix_ids = suffix_ids.clone()
        best_safe_prob = 0.0
        
        for step in tqdm(range(Config.NUM_STEPS), desc="Attack Steps"):
            
            # Decode current suffix
            current_suffix_str = self.tokenizer.decode(suffix_ids, skip_special_tokens=True)
            
            # Get input IDs with chat template
            input_ids, suffix_start, suffix_end = self.get_input_ids_with_suffix(
                malicious_prompt, current_suffix_str
            )
            
            # Adjust suffix slice based on actual tokenization
            # The suffix might not be exactly where we think due to tokenization
            # For now, use the estimated positions
            suffix_slice = slice(suffix_start, suffix_end)
            
            # Make sure slice is valid
            if suffix_start < 0 or suffix_end > len(input_ids):
                print(f"Warning: Invalid suffix slice [{suffix_start}:{suffix_end}] for input length {len(input_ids)}")
                # Fallback: assume suffix is near the end
                suffix_slice = slice(max(0, len(input_ids) - len(suffix_ids)), len(input_ids))
            
            # 1. Compute Gradients
            try:
                grad, loss, safe_prob = self.token_gradients(input_ids, suffix_slice)
            except Exception as e:
                print(f"Error in gradient computation: {e}")
                print(f"Input IDs length: {len(input_ids)}, Suffix slice: {suffix_slice}")
                continue
            
            # 2. Generate Candidates
            candidates = self.sample_control(suffix_ids, grad, Config.BATCH_SIZE, Config.TOP_K)
            
            # 3. Evaluate Candidates
            candidate_losses, candidate_safe_probs = self.evaluate_candidates(
                malicious_prompt, candidates
            )
            
            # 4. Select best candidate
            min_loss, best_idx = torch.min(candidate_losses, dim=0)
            
            if min_loss < best_loss:
                best_loss = min_loss.item()
                best_suffix_ids = candidates[best_idx].clone()
                best_suffix = self.tokenizer.decode(best_suffix_ids, skip_special_tokens=True)
                best_safe_prob = candidate_safe_probs[best_idx].item()
                suffix_ids = best_suffix_ids.clone()
                
            if step % 10 == 0:
                print(f"Step {step}: Loss = {best_loss:.4f}, Safe Prob = {best_safe_prob:.4f}, Suffix = '{best_suffix}'")

            if save_dir and save_interval and (step + 1) % save_interval == 0:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                save_path = os.path.join(save_dir, f"checkpoint_step_{step+1}.json")
                with open(save_path, "w") as f:
                    json.dump({
                        "step": step + 1,
                        "loss": best_loss,
                        "safe_prob": best_safe_prob,
                        "suffix": best_suffix,
                        "prompt": malicious_prompt,
                        "timestamp": datetime.datetime.now().isoformat()
                    }, f, indent=4)
        
        print(f"\n{'='*60}")
        print(f"Attack completed!")
        print(f"Best suffix: '{best_suffix}'")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Best safe probability: {best_safe_prob:.4f}")
        print(f"{'='*60}\n")
                
        return best_suffix, best_suffix_ids, best_loss, best_safe_prob
