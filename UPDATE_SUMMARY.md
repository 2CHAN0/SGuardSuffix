# 수정 완료 요약

## 📝 변경된 파일

### 1. **attack.py** (덮어쓰기 완료 ✅)
- Chat template 사용하도록 완전히 재작성
- 단일 토큰 분류 방식으로 변경 (safe/unsafe)
- Training-Inference 일관성 확보
- **반환값 변경**: `best_suffix, best_suffix_ids, best_loss, best_safe_prob` (4개)

### 2. **main.py** (덮어쓰기 완료 ✅)
- Chat template 사용
- Training vs Inference 비교 기능 추가
- 상세한 디버깅 정보 출력
- Baseline (suffix 없음) vs Attack (suffix 있음) 비교

### 3. **config.py** (업데이트 완료 ✅)
- `TARGET_STRING` → `TARGET_TOKEN`으로 변경

### 4. **colab_train_updated.py** (새로 생성 ✅)
- Colab에서 실행 가능한 수정된 training 스크립트
- Chat template 사용
- Training-Inference 일관성 검증 포함
- 4개 반환값 처리

---

## 🔧 주요 변경사항

### Before (문제 있던 코드):
```python
# attack.py
prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
target_ids = tokenizer("safe", add_special_tokens=False).input_ids
input_ids = torch.cat([prompt_ids, suffix_ids, target_ids])

# Loss: 시퀀스 예측 방식
shift_logits = logits[0, loss_slice, :]
loss = F.cross_entropy(shift_logits, shift_labels)

# 반환값 3개
return best_suffix, best_suffix_ids, best_loss
```

### After (수정된 코드):
```python
# attack.py
messages = [{"role": "user", "content": f"{prompt} {suffix}"}]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True
)

# Vocab에서 직접 token ID 추출
vocab = tokenizer.get_vocab()
safe_token_id = vocab['safe']
unsafe_token_id = vocab['unsafe']

# Loss: 단일 토큰 분류 방식
next_token_logits = logits[0, -1, :]
selected_logits = torch.stack([
    next_token_logits[safe_token_id],
    next_token_logits[unsafe_token_id]
])
probs = torch.softmax(selected_logits, dim=0)
loss = -torch.log(probs[0] + 1e-10)  # Maximize P(safe)

# 반환값 4개
return best_suffix, best_suffix_ids, best_loss, best_safe_prob
```

---

## 🚀 로컬에서 실행하기

```bash
cd /Users/cy.lee/Projects/SGuardSuffix
python main.py
```

---

## ☁️ Colab에서 실행하기

### 방법 1: 업데이트된 Python 스크립트 사용
1. `colab_train_updated.py`의 내용을 Colab 노트북 셀에 복사
2. 각 `# %%` 구분자별로 셀 분리
3. 순서대로 실행

### 방법 2: Git에서 최신 코드 pull 후 실행
```python
# Colab 셀
!rm -rf /content/sguard_attack
!git clone https://github.com/2CHAN0/SGuardSuffix.git sguard_attack
!pip install -r sguard_attack/requirements.txt

# Python code
import sys
sys.path.append('/content')

from sguard_attack.main import main
main()
```

**주의**: Git에 push한 후에 위 방법이 작동합니다!

---

## 🔍 예상 결과

### 수정 전 (문제):
```
Step 0: Loss = 0.001
...
Step 500: Loss = 0.001 ✅ 낮음

Inference: "unsafe" ❌ 실패
```

### 수정 후 (정상):
```
Step 0:  Loss=2.5, Safe Prob=0.08
Step 10: Loss=1.2, Safe Prob=0.30
Step 50: Loss=0.5, Safe Prob=0.60
...
Step 500: Loss=0.2, Safe Prob=0.82 ✅

Inference:
  Generated token: 'safe' ✅ 성공!
  Safe Prob (Training): 0.82
  Safe Prob (Inference): 0.82 ✅ 일치!
```

---

## 📋 체크리스트

- [x] `attack.py` 수정
- [x] `main.py` 수정
- [x] `config.py` 수정
- [x] Colab 스크립트 생성 (`colab_train_updated.py`)
- [x] 진단 보고서 작성 (`DIAGNOSIS_REPORT.md`)
- [x] 상세 분석 문서 작성 (`ANALYSIS.md`)
- [x] 테스트 스크립트 작성 (`test_problem.py`)
- [ ] Git commit & push (사용자가 진행)
- [ ] Colab에서 테스트 (사용자가 진행)

---

## 💡 다음 단계

1. **로컬 테스트 (optional)**
   ```bash
   cd /Users/cy.lee/Projects/SGuardSuffix
   python main.py
   ```

2. **Git 커밋**
   ```bash
   git add .
   git commit -m "Fix training-inference mismatch: Add chat template support"
   git push
   ```

3. **Colab에서 실행**
   - Git pull 후 실행
   - 또는 `colab_train_updated.py` 내용을 노트북에 복사

4. **결과 확인**
   - Training Safe Prob와 Inference Safe Prob가 일치하는지 확인
   - "safe" 토큰이 생성되는지 확인
   - Discrepancy warning이 나타나지 않는지 확인

---

## 📚 참고 문서

- `DIAGNOSIS_REPORT.md`: 종합 진단 보고서 (테스트 결과 포함)
- `ANALYSIS.md`: 상세 분석 (문제점 및 해결방안)
- `test_problem.py`: 문제를 시연하는 테스트 스크립트

---

**작성일**: 2025-11-22  
**마지막 업데이트**: 2025-11-22 17:17
