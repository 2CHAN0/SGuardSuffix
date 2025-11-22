# SGuard Suffix Attack 문제점 분석 및 해결방안

## 📋 발견된 문제점

### 🔴 문제 1: Chat Template 미사용 (가장 치명적)

**현재 코드 (잘못됨):**
```python
# attack.py, line 109
prompt_ids = self.tokenizer(malicious_prompt, return_tensors="pt", 
                           add_special_tokens=False).input_ids[0]
```

**올바른 방식:**
```python
messages = [{"role": "user", "content": malicious_prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    tokenize=True, 
    return_tensors='pt'
)
```

**왜 문제인가?**
- SGuard 모델은 chat template을 **반드시** 사용해야 합니다
- 공식 문서에서 `apply_chat_template()` 사용을 명시
- Training과 Inference에서 다른 형식을 사용하면 모델이 다르게 동작

**영향:**
- Training: 일반 텍스트 토큰화
- Inference: Chat template 사용 (main.py에서 `model.generate()` 호출 시)
- → 완전히 다른 입력 형식!

---

### 🔴 문제 2: 잘못된 Target Token 처리

**현재 코드:**
```python
# config.py
TARGET_STRING = "safe"

# attack.py, line 107
target_ids = self.tokenizer(target, return_tensors="pt", 
                           add_special_tokens=False).input_ids[0]
```

**올바른 방식:**
```python
vocab = tokenizer.get_vocab()
safe_token_id = vocab['safe']      # 단일 토큰 ID
unsafe_token_id = vocab['unsafe']  # 단일 토큰 ID
```

**왜 문제인가?**
- SGuard는 "safe" 또는 "unsafe" **단일 토큰**만 출력
- 토큰화하면 여러 토큰으로 쪼개질 수 있음
- vocab에서 직접 ID를 가져와야 정확

---

### 🔴 문제 3: 잘못된 Loss 계산

**현재 코드:**
```python
# attack.py, line 50-53
shift_logits = logits[0, loss_slice, :].contiguous()
shift_labels = input_ids[target_slice].contiguous()
loss = F.cross_entropy(shift_logits, shift_labels)
```

**문제점:**
- 여러 토큰에 대한 시퀀스 예측 loss 계산
- SGuard는 **첫 번째 생성 토큰**만 중요

**올바른 방식:**
```python
# 마지막 위치의 logit (다음 토큰 예측)
next_token_logits = logits[0, -1, :]

# safe와 unsafe 토큰의 logit만 추출
selected_logits = torch.stack([
    next_token_logits[safe_token_id],
    next_token_logits[unsafe_token_id]
])

# 확률 계산
probs = torch.softmax(selected_logits, dim=0)

# safe 토큰의 확률을 최대화
loss = -torch.log(probs[0] + 1e-10)
```

---

### 🔴 문제 4: Training-Inference 불일치

**Training (attack.py):**
```python
input_ids = torch.cat([prompt_ids, suffix_ids, target_ids])
# Chat template 없음, 그냥 concat
```

**Inference (main.py, line 31):**
```python
full_input = malicious_prompt + " " + best_suffix  # 공백 추가!
inputs = tokenizer(full_input, return_tensors="pt")
# 여전히 chat template 없지만, 공백이 추가됨
```

**Inference (main.py, line 46):**
```python
full_input_ids = torch.cat([prompt_ids, best_suffix_ids], dim=1)
# 공백 없음
```

**문제점:**
1. Training에서는 공백 없이 concat
2. Inference에서는 공백 있음 (line 31) 또는 없음 (line 46)
3. 둘 다 chat template을 사용하지 않음
4. 토큰화 결과가 달라짐

---

### 🔴 문제 5: 입력 구조의 개념적 오류

**현재 접근:**
```
[Prompt] + [Suffix] + [Target("safe")]
```

**왜 문제인가?**
- Target을 입력에 포함시키는 것은 일반 LM 학습 방식
- SGuard는 classification 모델
- Target은 **출력**이지 입력이 아님

**올바른 접근:**
```
Input:  [Prompt] + [Suffix] (chat template 적용)
Output: "safe" 또는 "unsafe" (단일 토큰)
Loss:   -log P(safe | input)
```

---

## ✅ 해결 방안

### 수정된 파일들

1. **attack.py**: 완전히 재설계된 GCG attack
   - Chat template 사용
   - 단일 토큰 분류 처리
   - 올바른 loss 계산
   - Training-Inference 일관성

2. **main.py**: 검증 로직 개선
   - Chat template 사용
   - Training vs Inference 비교
   - 상세한 디버깅 정보

3. **config.py**: TARGET_STRING → TARGET_TOKEN

### 핵심 변경사항

#### 1. Chat Template 사용
```python
messages = [{"role": "user", "content": f"{malicious_prompt} {suffix}"}]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors='pt'
)
```

#### 2. 올바른 Token ID 사용
```python
vocab = tokenizer.get_vocab()
self.safe_token_id = vocab['safe']
self.unsafe_token_id = vocab['unsafe']
```

#### 3. 올바른 Loss 계산
```python
next_token_logits = logits[0, -1, :]
selected_logits = torch.stack([
    next_token_logits[self.safe_token_id],
    next_token_logits[self.unsafe_token_id]
])
probs = torch.softmax(selected_logits, dim=0)
loss = -torch.log(probs[0] + 1e-10)  # Maximize P(safe)
```

#### 4. Training-Inference 일관성
- Training과 Inference 모두 동일한 방식 사용
- Chat template 적용
- 공백 처리 일관성

---

## 📊 예상 결과

### 이전 (문제 있는 코드):
```
Training Loss: 0.001  (낮음)
Inference Result: "unsafe"  (실패)
→ Loss는 낮지만 실제로는 작동하지 않음
```

### 수정 후:
```
Training Loss: 0.001
Inference Result: "safe"  (성공!)
→ Training과 Inference가 일치
```

---

## 🚀 사용 방법

### 기존 코드 (문제 있음):
```bash
python -m sguard_attack.main
```

### 수정된 코드:
```bash
python -m sguard_attack.main
```

---

## 🔍 추가 디버깅 팁

수정된 코드는 다음 정보를 출력합니다:

1. **Safe/Unsafe Token IDs**: 모델의 vocabulary에서 가져온 정확한 ID
2. **Training Safe Prob**: 학습 중 계산된 safe 확률
3. **Inference Safe Prob**: 실제 추론 시 safe 확률
4. **Discrepancy Warning**: 두 값의 차이가 0.1 이상이면 경고

이를 통해 Training-Inference 일관성을 실시간으로 확인할 수 있습니다.

---

## 📝 결론

**핵심 문제:**
- Chat template 미사용
- 잘못된 token 처리
- 잘못된 loss 계산
- Training-Inference 불일치

**해결책:**
- 모든 단계에서 chat template 사용
- Vocab에서 직접 token ID 추출
- 단일 토큰 classification에 맞는 loss
- 일관된 입력 형식

**결과:**
- Training loss와 inference 결과의 일치
- 실제로 "safe" 출력을 유도할 수 있는 suffix 생성
