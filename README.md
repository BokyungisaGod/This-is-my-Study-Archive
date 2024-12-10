📜: Paper link
🧑🏻‍💻: Developer blog & Github link & HuggingFace
🗞️: News
🤪: Interesting

---

# My study archive 2024

## 🎄 December
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
  
- 🧑🏻‍💻 https://chanmuzi.tistory.com/479
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
    
- 🗞️ [Introducing Motif: A High-Performance Open-Source Korean LLM by Moreh](https://moreh.io/blog/introducing-motif-a-high-performance-open-source-korean-llm-by-moreh-241202)
  - Moreh에서 한국어 성능이 뛰어난 초거대 언어 모델(LLM) 'Llama3-Motif-102B'을 오픈소스로 공개
    - 한국어 성능 강화를 위해 LlamaPro와 Masked Structure Growth(MSG) 등 최신 학습 기법을 활용해 개발
  - KMMLU 벤치마크에서 GPT-4를 능가하는 성적을 기록하였으며, Hugging Face와 GitHub에서 접근 가능
  - Llama 3 기반으로 MoAI 플랫폼을 활용하여 개발되었으며, 효율적 GPU 관리 및 모델 최적화 가능
  - 향상된 한국어 처리 능력과 영어 성능을 동시에 제공
  - [테스트 해보기](https://model-hub.moreh.io/text)

- 📜 [IST, ETH] [GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS](https://arxiv.org/abs/2210.17323)
  - 기존 GPT 모델의 한계 : 모델 크기가 방대하여 추론에 많은 GPU가 필요해 실용성이 떨어짐
  - GPTQ: GPT모델의 높은 컴퓨팅 및 저장 비용 문제를 해결하기 위한 새로운 양자화 방법
    <details>
        <summary>주요 특징</summary>
      
      - 원샷 가중치 양자화: 한 번의 과정을 통해 모델의 가중치를 효율적으로 압축
      - 1750억 개 파라미터를 가진 GPT 모델을 약 4시간 만에 양자화 가능
      - 가중치당 비트 폭을 3~4비트로 줄여도 성능 저하가 거의 없음
      - 기존 양자화 기법 대비 2배 이상 효율적
    </details>
    <details>
        <summary>주요 성과 및 추론 속도 향상</summary>
      
      - 1750억 개 파라미터 모델도 단일 GPU로 처리 가능
      - FP16 대비 추론 속도
        - 고급 GPU(NVIDIA A100): 3.25배,
        - 비용 효율적인 GPU(NVIDIA A6000): 4.5배 빨라짐
      - 극한 양자화에서도 정확도 유지
        - 가중치를 2비트 또는 3진수로 줄여도 합리적인 성능 유지 
    </details>
  - [Github](https://github.com/IST-DASLab/gptq)

- 🗞️ [Google DeepMind] [Genie 2: A large-scale foundation world model](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
  - Genie 2: 이미지 하나만으로 다양한 3D 환경을 생성하는 기반 세계 모델
    - 사람 또는 AI 에이전트가 키보드와 마우스로 조작하며 상호 작용할 수 있는 무한한 가상 환경을 제공
    - 물리 효과, 캐릭터 애니메이션, 객체 상호 작용 등을 모델링하여 현실적인 가상 세계를 생성하며, 실제 이미지를 기반으로 한 환경 생성도 가능
    - AI 에이전트 훈련 및 평가에 유용한 다양한 환경을 빠르게 제작하는 데 활용

- 📜 [KU, KAIST] [CheckEval: Robust Evaluation Framework using Large Language Model via Checklist](https://arxiv.org/abs/2403.18771)
  - CheckEval: LLM을 활용한 새로운 평가 프레임워크로, 기존 평가 방법의 문제점(모호한 평가 기준, 불일치)을 개선하기 위해 설계
    <details>
        <summary>해결 방법</summary>
          
      - 평가 기준을 세부적인 하위 측면으로 나눔
      - 각 측면별로 Boolean 질문 체크리스트를 만들어 평가 과정을 단순화
      - 해석 가능성 높임, 특정 평가 항목에 초점 → 결과의 견고성, 신뢰성 강화
    </details>

    <details>
        <summary>주요 성과</summary>
          
      - SummEval 벤치마크를 활용한 집중 사례 연구 → CheckEval: 인간의 판단과 높은 상관관계를 보임
      - IAA(Inter-Annotator Agreement)가 매우 높음
      - 객관적이고 유연하며 정밀한 평가에 효과적임을 입증
    </details>

    <details>
        <summary>중요 개념</summary>
      
      - CheckEval: 평가의 명확성과 일관성을 높이기 위해 설계된 LLM 기반 평가 프레임워크
      - Inter-Annotator Agreement (IAA): 평가자 간의 일치도를 측정하는 지표
      - SummEval : 요약에 대한 다양한 평가 방법을 비교하는 벤치마크 데이터셋
    </details>

- 📜 [Google] [PaliGemma 2: A Family of Versatile VLMs for Transfer](https://arxiv.org/abs/2412.03555)
  - PaliGemma 2: 기존 PaliGemma 모델을 기반으로 업그레이드된 VLM으로, Gemma 2 언어 모델 계열의 개선된 기능을 통합한 모델
  - Gemma 2 언어 모델 계열(2B ~ 27B 파라미터)과 SigLIP-So400m 비전 인코더 통합

    <details>
        <summary>3가지 해상도(224px², 448px², 896px²)에서 다단계 훈련</summary>
      
      - 전이 학습 능력 강화, 세부 조정 가능
      - 학습률, 작업 유형, 모델 크기, 해상도 등 전이 성능 영향 요소 분석
    </details>

    <details>
        <summary>작업 범위 확장</summary>
    
      - OCR 관련 작업: 테이블 구조, 분자 구조, 악보 인식
      - 세밀한 장기 캡션 생성, 방사선 보고서 작성
      - 다양한 전이 작업에서 최첨단 성능(SOTA) 달성
    </details>

    <details>
        <summary>중요 개념</summary>
  
      - Vision-Language Model (VLM): 이미지를 처리하는 비전 모델과 텍스트를 이해하는 언어 모델을 결합한 AI 모델
      - 전이 학습(Transfer Learning): 이미 학습된 모델을 새로운 작업에 적응시키는 방법
    </details>

  - [HuggingFace](https://huggingface.co/papers/2412.03555), [Kaggle](https://www.kaggle.com/models/google/paligemma-2)
</details>
  
<details>
  <summary>2nd week</summary>

- 🧑🏻‍💻 [NVIDIA] [Content Moderation and Safety Checks with NVIDIA NeMo Guardrails](https://developer.nvidia.com/blog/content-moderation-and-safety-checks-with-nvidia-nemo-guardrails/)
  - RAG application: 실시간으로 외부 데이터를 검색하고 LLM을 활용하여 동적인 콘텐츠를 생성
    - 안전하고 신뢰할 수 있는 응답을 보장하기 위해 content moderation 필수적
  - NVIDIA NeMo Guardrails: LLM의 입력 및 출력을 관리하는 toolkit/microservice

     <details>
         <summary>주요 기능</summary>
       
       - LlamaGuard
          - 입력/출력에서 부적절한 콘텐츠 감지
       - AlignScore
          - 응답의 사실 검증(검색 데이터와 생성된 결과 비교)

       - 기타 기능: 식별 정보(PII) 검출, 허위 정보 방지, 탈옥 감지 등
    </details>

    <details>
        <summary>적용 방법</summary>

      - NeMo Guardrails를 설치
      - RAG 애플리케이션과 연동
      - LlamaGuard 및 AlignScore 모델을 설정
      - NeMo Guardrails의 구성 파일(config.yml)에 통합
      - 보안 레이어를 구성하고 샘플 쿼리로 테스트
    </details>

- 🤪 [ElevenLabs](https://www.talktosanta.io/)

- 🤪 [Microsoft] [MicrosoftDesigner](https://designer.microsoft.com/design)

- 🧑🏻‍💻 [Docling] [Docling](https://ds4sd.github.io/docling/)
  - PDF, DOCX, PPTX 등 다양한 문서 형식을 읽어 Markdown 및 JSON 형식으로 변환하는 도구
  - 페이지 레이아웃, 읽기 순서, 표 구조 등을 포함한 고급 PDF 문서 이해 기능과 🦙 LlamaIndex, 🦜🔗 LangChain과의 쉬운 통합 제공
  - OCR 지원, CLI 제공 등 사용 편의성을 높였으며, 추후 방정식 및 코드 추출, 메타데이터 추출 기능 추가 예정

- 🤪 [DVC] [DVC](https://dvc.org/)
  - DVC(Data Version Control): GitOps 원칙에 기반하여 대규모 데이터의 버전 관리 및 ML 모델링 프로세스의 재현 가능한 워크플로우 구축을 지원하는 오픈소스 플랫폼
  - [Github](https://github.com/iterative/dvc)

- 🧑🏻‍💻 [Meta] [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
  - Meta Llama 3.3: Meta가 개발한 70B 파라미터 규모의 다국어 지원 LLM으로, 사전 학습과 명령어 조정을 통해 다국어 대화, 자연어 생성, 코딩 지원 등 다양한 사용 사례에 최적화
    <details>
        <summary>모델 아키텍처</summary>

      - 트랜스포머 기반: 최적화된 트랜스포머 아키텍처를 활용한 자동 회귀 모델
      - 명령어 조정: 감독 미세 조정(SFT)과 인간 피드백 기반 강화 학습(RLHF)을 통해 사용자의 도움 및 안전성 선호도에 맞게 조정됨
    </details>

    <details>
        <summary>벤치마크</summary>
    
      - MMLU(CoT): 86.0% 정확도
      - HumanEval: 88.4% 성공률
      - MATH(CoT): 77.0% 정확도
      - MGSM: 91.1% 정확도
    </details>

</details>
