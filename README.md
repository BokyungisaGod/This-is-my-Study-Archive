📜: Paper link
🧑🏻‍💻: Developer blog & Github link
🗞️: News
📝: Study tip

---

# My study archive 2024

## ⛄️ December
<details>
  <summary>1st week</summary>

- 📜 [Harvard, Stanford, MIT, Databricks, CMU] [Scaling Laws for Precision](https://arxiv.org/pdf/2411.04330)
    - 낮은 정밀도(Low precision)로 학습과 추론을 수행할 때의 영향을 연구했으며, 이를 예측하는 새로운 스케일링 법칙 제시
      - 학습 시: 낮은 정밀도는 모델의 유효 파라미터 수를 감소시키는 효과가 있음을 발견
      - 추론 시: 데이터가 많아질수록 양자화로 인한 성능 저하가 커져서, 오히려 추가 사전학습이 해로울 수 있음
    - 특히 대규모 모델의 경우 저정밀도 훈련이 계산 효율성 측면에서 최적일 수 있다는 점 제시
    - 1.7B 파라미터 규모의 모델과 26B 토큰 데이터셋으로 검증하여, 학습과 추론 시의 정밀도 변화에 따른 성능 저하를 예측하는 통합된 수식 제시
  <details>
    <summary>중요 개념</summary>
    
    - precision(정밀도): 숫자를 얼마나 정확하게 표현하는지의 정도
    - scaling laws(스케일링 법칙): 모델의 크기와 성능 관계를 설명하는 규칙
    - quantization(양자화): 데이터를 더 작은 비트로 압축하는 과정
  </details>
  
- 📝 https://chanmuzi.tistory.com/479
  - NLP, LLM 위주의 인공지능 최신 논문/뉴스 follow-up 팁

- 📜 [RAPID RESPONSE: MITIGATING LLM JAILBREAKS WITH A FEW EXAMPLES](https://arxiv.org/abs/2411.07494)
  - LLM의 안전성 확보를 위해, 완벽한 방어가 아닌 신속 대응 기법에 초점을 맞춤
  - 소수의 공격 사례만으로도 유사한 형태의 전체 공격 유형을 차단하는 방법 제시
  → 이를 평가하기 위한 'RapidResponseBench' 벤치마크 개발
    <details>
      <summary>'<b>탈옥 확산(jailbreak proliferation)</b>' 기반의 5가지 방어 기법 평가</summary>
      
    - 관찰된 공격 사례를 바탕으로 자동으로 유사한 jailbreak를 생성하여 방어에 활용
    - 가장 효과적인 방법: 생성된 jailbreak를 차단하도록 입력 분류기를 미세조정
    - 단 하나의 공격 사례만으로도 동일 유형 공격은 1/240, 새로운 유형 공격은 1/15로 성공률 감소
    </details>
  
  - 추가 연구를 통해 방어 효과에 영향을 미치는 핵심 요소 파악
    <details>
      <summary>중요 역할</summary>
  
    - 확산 모델의 품질:  생성된 탈옥 사례의 다양성과 적합성
    - 생성된 탈옥 사례 수: 더 많은 사례가 더 강력한 방어로 이어짐
    </details>
  
</details>
