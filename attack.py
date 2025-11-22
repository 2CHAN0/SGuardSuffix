import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from .config import Config

class GCGAttack:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = Config.DEVICE
        
    def token_gradients(self, input_ids, suffix_slice, target_slice, loss_slice):
        """
        Computes gradients of the loss with respect to the one-hot text input
        at the suffix positions.
        """
        # Create embeddings matrix
        embed_weights = self.model.get_input_embeddings().weight
        
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
        one_hot.requires_grad_()
        
        # Get embeddings
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        
        # Forward pass
        outputs = self.model(inputs_embeds=input_embeds)
        logits = outputs.logits
        
        # Calculate loss
        # We want to maximize the probability of the target string
        # Target is "safe"
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[0, loss_slice, :].contiguous()
        shift_labels = input_ids[target_slice].contiguous()
        
        loss = F.cross_entropy(shift_logits, shift_labels)
        
        # Backward pass
        loss.backward()
        
        # Get gradients for the suffix tokens
        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)
        
        return grad[suffix_slice]

    def sample_control(self, control_toks, grad, batch_size, topk=256):
        """
        Randomly sample candidate tokens based on gradients.
        """
        control_toks = control_toks.to(self.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=self.device
        ).type(torch.int64)
        
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

    def run(self, malicious_prompt):
        """
        Runs the GCG attack for a single malicious prompt.
        """
        # Initialize suffix
        suffix = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !" # 20 tokens
        suffix_ids = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False).input_ids[0].to(self.device)
        
        target = Config.TARGET_STRING
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids[0].to(self.device)
        
        prompt_ids = self.tokenizer(malicious_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].to(self.device)
        
        # Define slices
        # Input structure: [Prompt] [Suffix] [Target]
        # We want to optimize [Suffix] such that P([Target] | [Prompt] [Suffix]) is maximized
        
        best_loss = float('inf')
        best_suffix = suffix
        
        for step in tqdm(range(Config.NUM_STEPS), desc="Attack Steps"):
            
            # Construct full input sequence for gradient computation
            # Note: We don't include target in input for generation, but we need it for loss calculation
            # For CausalLM, we feed [Prompt] [Suffix] and calculate loss on [Target]
            
            # Actually, for gradient computation we need to feed everything including target to get logits for target positions
            # But we only want gradients w.r.t suffix
            
            input_ids = torch.cat([prompt_ids, suffix_ids, target_ids])
            
            # Slices indices
            p_len = len(prompt_ids)
            s_len = len(suffix_ids)
            t_len = len(target_ids)
            
            suffix_slice = slice(p_len, p_len + s_len)
            target_slice = slice(p_len + s_len, p_len + s_len + t_len)
            loss_slice = slice(p_len + s_len - 1, p_len + s_len + t_len - 1) # Logits are shifted by 1
            
            # 1. Compute Gradients
            grad = self.token_gradients(input_ids, suffix_slice, target_slice, loss_slice)
            
            # 2. Generate Candidates
            candidates = self.sample_control(suffix_ids, grad, Config.BATCH_SIZE, Config.TOP_K)
            
            # 3. Evaluate Candidates
            # Create batch of inputs
            # [Batch, Seq_Len]
            # Each row: [Prompt] [Candidate_Suffix] [Target]
            
            # We need to be careful with memory here.
            # Let's evaluate in mini-batches if needed, but for now assume it fits.
            
            # Construct batch
            # prompt_ids: [P] -> [B, P]
            # candidates: [B, S]
            # target_ids: [T] -> [B, T]
            
            b_prompts = prompt_ids.repeat(Config.BATCH_SIZE, 1)
            b_targets = target_ids.repeat(Config.BATCH_SIZE, 1)
            
            b_input_ids = torch.cat([b_prompts, candidates, b_targets], dim=1)
            
            with torch.no_grad():
                outputs = self.model(b_input_ids)
                logits = outputs.logits
                
                # Calculate loss for each candidate
                # Logits: [B, Seq_Len, Vocab]
                # We care about logits at loss_slice positions matching target_ids
                
                # shift_logits: [B, T, Vocab]
                shift_logits = logits[:, loss_slice, :].contiguous()
                # shift_labels: [B, T]
                shift_labels = b_targets
                
                # Flatten for cross_entropy
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none'
                )
                
                # Reshape to [B, T] and sum/mean over T
                loss = loss.view(Config.BATCH_SIZE, -1).mean(dim=1)
                
                # Find best candidate
                min_loss, best_idx = torch.min(loss, dim=0)
                
                if min_loss < best_loss:
                    best_loss = min_loss.item()
                    best_suffix_ids = candidates[best_idx]
                    best_suffix = self.tokenizer.decode(best_suffix_ids)
                    suffix_ids = best_suffix_ids
                    
            if step % 10 == 0:
                print(f"Step {step}: Loss = {best_loss:.4f}, Suffix = {best_suffix}")
                
        return best_suffix, best_loss
