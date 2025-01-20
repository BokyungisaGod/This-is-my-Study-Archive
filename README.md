📜: Paper link
🧑🏻‍💻: Developer blog & Github link & HuggingFace
🗞️: News
🤪: Interesting

---
# My study archive 2025

## 🎉 January
<details>
  <summary>1st week</summary>

  - 🧑🏻‍💻 [instructkr] [retriever-simple-benchmark](https://github.com/instructkr/retriever-simple-benchmark)
    - Instructkr팀이 제작한 retriever-simple-benchmark의 결과를 보여주는 GitHub 저장소
    - 다양한 검색 시스템의 성능 비교를 위한 벤치마크 결과를 담고 있음
      <details>
          <summary>중요 개념</summary>
        
        - **retriever-simple-benchmark**: RAG에 필요한 리랭커를 평가하기 위해 설계된 가볍고 효율적인 벤치마크 프로젝트
      </details>

  - 🧑🏻‍💻 [ollama] [kwangsuklee/llama3.2-3B-Q8-korean](https://ollama.com/kwangsuklee/llama3.2-3B-Q8-korean)
    - llama-3.2-3B-Q8-korean: 3.2B 파라미터를 가지는 한국어 모델로, Q8_0 quantization 방식 사용
      - 모델 생성 과정: Hugging Face의 Bllossom/llama-3.2-Korean-Bllossom-3B 모델을 기반
    - [Github](https://github.com/ollama/ollama)
    - [HuggingFace] [Bllossom/llama-3.2-Korean-Bllossom-3B](https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B)
      - 텍스트 기반의 한국어-영어 강화 언어모델
    - [HuggingFace] [Bllossom/llama-3.2-Korean-Bllossom-AICA-5B](https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-AICA-5B)
      - upgrade ver.
      - 텍스트와 이미지 모두 처리 가능한 시각-언어모델
      - 특히 OCR, 표, 그래프 해석 등 시각 정보 처리에 최적화
  
  - 🧑🏻‍💻 [sarthakrastogi] [quality-prompts](https://github.com/sarthakrastogi/quality-prompts/tree/main)
    - 58가지 프롬프트 기법을 구현한 라이브러리
    - 사용자 질의와 관련된 소수의 예시만 검색하여 사용, 문맥 명확화 및 단계별 사고 과정을 통한 정확도 향상을 위한 기능(system2attention, tabular_chain_of_thought_prompting)제공
      <details>
          <summary>중요 개념</summary>
        
      - **System2Attention**: Transformer 모델의 Attention 메커니즘을 확장하여 논리적 추론과 복잡한 문제 해결을 지원하는 방식
      - **Tabular Chain of Thought Prompting**: 테이블 데이터를 단계적으로 추론하도록 유도해 모델이 열과 행 간 관계를 분석하며 답을 도출하게 하는 방법
      </details>
  - 🧑🏻‍💻 [Medium][Guidebook to the State-of-the-Art Embeddings and Information Retrieval](https://sigridjin.medium.com/rag-%EC%84%B8%EC%83%81%EC%9D%84-%ED%97%A4%EC%97%84%EC%B9%98%EB%8A%94-%EC%82%AC%EB%9E%8C%EB%93%A4%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B0%80%EC%9D%B4%EB%93%9C%EB%B6%81-3e90f515d800)
    - 최신 임베딩 및 정보 검색 기술에 대한 가이드북
    - 저자의 ColBERT와 Vespa 활용 실험 결과와 다양한 모델 평가, 하이브리드 검색 시스템의 장점 등을 다룸
    - 단일 임베딩 모델의 일반화 성능 한계를 지적하며, BM25와 같은 전통적 검색 기법과의 결합을 통한 하이브리드 시스템의 효용성을 강조하고, BGE-M3 등 다양한 모델의 성능 비교 및 양자화, 최적화 기법 제시
    - ColBERT를 활용한 해석 가능한 신경망 검색 구현 방법 소개, 토큰 단위 점수 확인을 통한 검색 결과의 신뢰도 향상 및 RAG 시스템 개선 방안 제시, 오픈소스 기반의 유연하고 효율적인 정보 검색 시스템 구축의 중요성 강조
      <details>
          <summary>중요 개념</summary>
        
        - **Embedding**: 데이터(텍스트, 이미지 등)를 고차원 공간에 벡터로 표현하는 기법
        - **ColBERT**: 토큰 단위의 세밀한 유사도 계산을 지원하는 신경망 기반 검색 모델
        - **BM25**: 텍스트 기반 검색을 위한 전통적 가중치 계산 알고리즘
        - **Hybrid Search System**: 전통적 검색 기법과 신경망 기반 검색 기법을 결합한 검색 시스템
        - **BGE-M3**: 특정 임베딩 기반 검색 모델
      </details>

  - 🧑🏻‍💻 [Byaidu] [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
    - AI 기반으로 PDF 과학 논문을 번역하며, 수식, 차트, 목차, 주석 등의 서식을 최대한 유지
    - Google, DeepL, Ollama, OpenAI 등 다양한 번역 서비스 지원
    - 다국어 지원, 다중 스레드 번역, 사용자 정의 프롬프트, 출력 디렉토리 지정 등 다양한 옵션 제공

  - 🧑🏻‍💻 [Ditto_GPT님 tistory] [범용 프롬프트 모음-Custom instructions 에 넣을 프롬프트 귀찮으면 그냥 이것만 쓰세요](https://cprompters.tistory.com/71)
    - ChatGPT의 Custom Instructions에 넣을 수 있는 범용 프롬프트 모음
      - 작업 우선순위 설정, 단계별 사고 과정 안내, 목표 명확화 및 세분화, 불필요한 설명 생략 등의 기능 포함

  - 📜 [NVIDIA, HuggingFace] [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)
    - ModernBERT: 기존 BERT와 같은 인코더 전용 변환기 모델을 최적화하여 Pareto 개선(성능-크기 tradeoff)을 달성한 모델  
      - 검색(단일 및 다중 벡터)과 분류 작업 등에서 뛰어난 평가 결과 기록, 코드 도메인에서도 검증됨  
      - 빠른 속도, 메모리 효율성, 일반 GPU에서의 추론에 최적화

  - 🧑🏻‍💻 [루닥스님 tistory] [langgraph-ReAct AgentExecutor in LangGraph](https://rudaks.tistory.com/451)
    - Langchain과 Langgraph를 활용하여 서울 날씨 정보를 얻고 3배수하는 ReAct Agent를 만드는 과정을 설명
    - `TavilySearchResults` tool을 통해 날씨 정보를 가져오고, `triple` tool을 통해 3배수 연산 수행

  - 🧑🏻‍💻 [jungjun hur님 velog] [앤트로픽, OpenAI, LangChain 팀의 LLM 에이전트](https://velog.io/@shangrilar/%EC%95%A4%ED%8A%B8%EB%A1%9C%ED%94%BD-OpenAI-LangChain-%ED%8C%80%EC%9D%98-LLM-%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8)
    - LLM 에이전트의 정의
      - Anthropic: 워크플로우 방식에 따라 구분
      - LangChain: LLM이 애플리케이션 흐름을 결정하는 시스템
      - OpenAI: 도구와 결합된 루틴

  - 🧑🏻‍💻 [School of Mechanical Engineering, Beijing Institute of Technology] [Large Language Model guided Deep Reinforcement Learning for Decision Making in Autonomous Driving](https://bitmobility.github.io/LGDRL/)
    - LGDRL: LLM 기반 심층 강화 학습 프레임워크
      - LLM이 DRL 학습에 지능적 가이드 제공, 90% 성공률 달성  
      - 가이드 없이도 안정적인 성능 유지, 실제 적용 가능성 증대
      - 전문가 정책 제약 알고리즘과 LLM 상호작용으로 성능 및 효율 극대화
    - [Github](https://github.com/bitmobility/LGDRL)
    - 📜 [Large Language Model guided Deep Reinforcement Learning for Decision Making in Autonomous Driving](https://arxiv.org/abs/2412.18511)
      <details>
          <summary>중요 개념</summary>
        
        - **DRL(Deep Reinforcement Learning)**: 심층 신경망을 활용하여 에이전트가 환경과 상호작용하며 최적의 행동 정책을 학습하는 강화 학습 기법
        - **전문가 정책 제약 알고리즘**: 전문가의 정책을 가이드로 삼아 에이전트의 학습 과정을 제약하거나 보조하여 성능을 향상시키는 알고리즘
      </details>

  - 🧑🏻‍💻 [microsoft] [markitdown](https://github.com/microsoft/markitdown)
    - MarkItDown: PDF, PowerPoint, Word, Excel 등 다양한 파일 형식을 Markdown으로 변환하는 파이썬 기반 유틸리티
      - 이미지, 오디오 파일 지원 및 LLM을 활용한 이미지 설명 기능 제공, 여러 파일을 일괄 처리하는 기능 제공
  
  - 🧑🏻‍💻 [Msty] [The easiest way to use local and online AI models](https://msty.app/)
    - Msty: 로컬 및 온라인 AI 모델을 간편하게 사용할 수 있는 애플리케이션
      - 다양한 모델(Hugging Face, Ollama, Open Router 등)과의 호환성 제공
      - 개인 정보 보호 및 안정성 보장(오프라인 우선 설계), 병렬 대화 기능, 지식 스택 기능
  
  - 🧑🏻‍💻 [sionic-ai] [2024-responsible-ai-in-action-gdsc-example](https://github.com/sionic-ai/2024-responsible-ai-in-action-gdsc-example)
    - ModernBERT를 활용하여, 사용자 질의에 적합한 LLM을 라우팅하는 300M 크기의 BERT 분류기를 구현하는 실습 과정
    - [ModernBERT 공식 문서] [Fine-tune classifier with ModernBERT in 2025](https://www.philschmid.de/fine-tune-modern-bert-in-2025)
    - [HuggingFace 트랜스포머 문서] [Finally, a Replacement for BERT](https://huggingface.co/blog/modernbert)
  
  - 🧑🏻‍💻 [TILNOTE] [Mastering Machine Translation: Understanding the Transformer Model through "Attention Is All You Need"](https://tilnote.io/pages/67749c26ff6e2b1f363b645f)
    - 순환 및 합성곱 연산 없이 어텐션 메커니즘만을 사용하는 새로운 신경망 아키텍처인 Transformer 모델 제시
    - 기계 번역에서 뛰어난 성능, 병렬 처리 효율성을 보여줌
    - 인코더-디코더 구조와 셀프 어텐션 메커니즘을 통해 시퀀스 데이터를 효과적으로 처리, 다양한 NLP 작업(텍스트 요약, 질문응답 등)에 적용 가능
    - 📜 [Google] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
      <details>
          <summary>중요 개념</summary>
        
        - **순환(Recurrent)**: 이전 상태의 출력을 현재 입력과 함께 사용해 시퀀스 데이터를 처리하는 구조, RNN이 대표적
        - **합성곱(Convolution)**: 입력 데이터의 지역적인 특징을 추출하기 위해 필터를 사용하는 연산, CNN에서 주로 사용
      </details>

  - 🧑🏻‍💻 [Microsoft, Health and Life Sciences AI] [MEDEC: A BENCHMARK FOR MEDICAL ERROR DETECTION AND CORRECTION IN CLINICAL NOTES](https://arxiv.org/pdf/2412.19260)
    - LLMs는 의료 질문에 정확히 답하지만, 기존 의료 텍스트의 오류 검증 및 수정 능력에 대한 연구가 부족
    - MEDEC1: 진단, 관리, 치료 등 5가지 오류 유형을 포함한 최초의 의료 오류 벤치마크
      - 오류 탐지에서 성과를 보였으나, 여전히 의료 전문가보다는 낮은 성능 → 성능 격차 원인 분석 및 평가 지표 개선 필요

  - 🧑🏻‍💻 [Prompt Engineering Guide](https://www.promptingguide.ai/)
    - LLM을 효과적으로 활용하기 위한 프롬프트 설계와 최적화 방법을 다룬 가이드
</details>
<details>
  <summary>2nd week</summary>

  - 📜 [Department of Computer Science National Chengchi University] [Don’t Do RAG:When Cache-Augmented Generation is All You Need for Knowledge Tasks](https://arxiv.org/abs/2412.15605)
    - RAG의 문제 → 검색 지연, 문서 선택 오류, 시스템 복잡성 증가
    - CAG → 긴 문맥 창을 가진 LLM의 특성을 활용해 실시간 검색 없이 사전 로드된 데이터를 사용
      - 필요한 지식이나 문서를 모델의 문맥 창에 미리 로드하고, 런타임 매개변수를 캐싱하여 검색 단계 제거
      - 검색 지연과 오류를 없앰 + 문맥 적합성 유지
      - 일부 벤치마크에서 RAG보다 뛰어난 성능을 보임
  - 🧑🏻‍💻 [HAMA님 tistory][정규표현식(Regex)정리](https://hamait.tistory.com/342)
    - 정규 표현식(Regex)의 주요 메타 문자(^, $, ., +, ?, *, |, () 등)와 의미, 플래그(g, i, m)의 기능을 설명
    - 특수 메타 문자([], [^], [x-z], \ 뒤의 문자들)의 사용법과 패턴 매칭 예시 알려줌

  - 📜 [Fudan University,Shanghai AI Laboratory][Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective](https://arxiv.org/abs/2412.14135)
    - OpenAI o1: 강력한 추론 능력을 통해 여러 복잡한 과제에서 전문가 수준의 성능을 달성한 인공지능 모델
      <details>
          <summary>주요 기술적 기반 → 강화 학습(정책 초기화, 보상 설계, 탐색, 학습)</summary>
          
        - 정책 초기화: 인간과 유사한 추론 행동을 학습하여 복잡한 문제 해결 능력 제공
        - 보상 설계: 탐색과 학습을 위한 효과적인 지침 역할을 하는 보상 신호 제공
        - 탐색: 훈련과 테스트 단계에서 더 나은 솔루션 생성
        - 학습: 탐색 데이터를 활용해 성능 개선
      </details>
  - 🧑🏻‍💻 [HuggingFace] [unsloth/DeepSeek-V3-GGUF](https://huggingface.co/unsloth/DeepSeek-V3-GGUF)
    - Unsloth의 DeepSeek-V3-GGUF: Llama 3.3, Gemma 2, Mistral을 최대 5배 빠르게, 메모리 사용량은 70% 줄여 미세 조정하는 오픈소스 모델
      - 다양한 양자화 버전(2~8bit)과 GGUF 형식을 지원
      - 다양한 벤치마크(MMLU, HumanEval 등)에서 우수한 성능을 보이며, 특히 수학 및 코드 관련 작업에서 강점
      - 최대 128K의 컨텍스트 윈도우 지원, DeepSeek 공식 웹사이트(chat.deepseek.com)와 API 플랫폼(platform.deepseek.com)을 통해 채팅 및 API 접근 가능
    - 📜 [DeepSeek-AI] [DeepSeek-V3 Technical Report](arxiv.org/abs/2412.19437)
      - DeepSeek-V3: 671B 파라미터를 가진 MoE 언어 모델, 각 토큰에 37B 파라미터를 활성화 → 효율적, 비용 효과적인 학습&추론 제공
      - MLA과 DeepSeekMoE 아키텍처 활용, 보조 손실 없이 로드 밸런싱과 멀티 토큰 예측 훈련 목표를 도입

  - 🧑🏻‍💻 [Sionic AI] [Sionic AI](https://blog.sionic.ai/)
    - Sionic AI에서 제공하는 기술 블로그
      - 기계학습, 딥러닝, 자연어 처리 등 다양한 AI 관련 주제의 글들 제공
      - 주요 내용 → RAG를 활용한 GPT 활용법, BGE-M3 모델 구현, LLM 평가 및 개선 방법, 효과적인 프롬프팅 기법 등 실용적인 AI 기술과 최신 연구 동향

  - 🧑🏻‍💻 [VITA] [VITA-MLLM VITA](https://github.com/VITA-MLLM/VITA)
    - VITA-1.5: 실시간 시각 및 음성 상호작용에서 GPT-4o 수준을 목표로 하는 다중 모드 대규모 언어 모델
      - 음성 처리 성능 향상 및 이미지 이해 성능 유지 달성
      - 이미지, 비디오, 음성 데이터를 포함하는 대규모 데이터셋을 사용하여 훈련됨
      <details>
          <summary>VITA-1.5의 훈련</summary>
        
        - InternViT-300M-448px, 사전 훈련된 오디오 인코더 등의 요소가 필요
        - 제공된 `finetuneTaskNeg_qwen_nodes.sh` 스크립트를 사용하여 지속적인 학습 가능
      </details>
    - [Demo Video](https://www.youtube.com/watch?v=tyi6SVFT5mM&ab_channel=BradyFU)

  - 🧑🏻‍💻 [CodeCrafters] [CodeCrafters](https://codecrafters.io/)
    - CodeCrafters: 실제 프로젝트를 통해 고급 프로그래밍 실력 향상을 돕는 플랫폼
      - Redis, Git, SQLite 등을 직접 구현하는 과제 제공
      - 자신의 IDE와 Git을 사용하여 코딩하고 실시간 피드백 받기 가능, 단순한 CRUD 기능이 아닌 실제 동작하는 소프트웨어를 구현하는 과제들 제공
</details>
<details>
    <summary>3rd week</summary>
  
  - 🧑🏻‍💻 [HuggingFace] [mistralai/Mistral-Small-Instruct-2409](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409)
    - Mistral-Small-Instruct-2409: 220억 개의 파라미터를 가진 Instruct Fine-Tuning(지시 사항 미세 조정) 버전 언어 모델로, 최대 32k 토큰 길이 처리
      - 단일 GPU에서 실행하려면 최소 44GB의 GPU 메모리 필요
      - 사용자 질문에 대한 답변 생성 및 작업 수행에 최적화
      - vLLM 라이브러리를 사용하여 효율적인 추론 파이프라인을 구축하는 것이 권장됨

  - 📜 [AMD, Johns Hopkins University] [Agent Laboratory: Using LLM Agents as Research Assistants](https://arxiv.org/abs/2501.04227)
    - Agent Laboratory: LLM 기반 프레임워크로, 문헌 검토, 실험, 보고서 작성 등 전체 연구 과정을 자동으로 처리
      - 연구 아이디어를 입력하면 연구자가 각 단계에서 피드백과 지침을 제공하며 결과 개선 가능
      - 연구 비용 → 이전 자율 연구 방법 대비 84% 절감
      - o1-preview로 구동된 Agent Laboratory가 가장 우수한 연구 결과 생성

  - 🧑🏻‍💻 [Channy님 블로그] [2025년 생성형 AI 기술 및 시장 요약](https://channy.creation.net/blog/1905)
    - 2025년 생성형 AI 시장 → 중소형 모델 중심으로 전환되며, 오픈소스 모델도 활발히 출시될 전망
    - 기업들은 보안, 가격, 성능, 확장성을 고려해 멀티 모델 전략을 채택하고, RAG 방식과 벡터 데이터베이스 활용이 주류를 이룰 것
    - 생성형 AI는 코딩 지원, 챗봇, 엔터프라이즈 검색 등 다양한 분야에서 활용될 예정
    - 에이전트 기반 시스템과 비용 절감 전략이 중요해지며, 코딩 도우미 시장의 경쟁이 심화될 전망
   
  - 🧑🏻‍💻 [buriy] [python-readability](https://github.com/buriy/python-readability)
    - arc90의 Readability 도구를 기반으로 제작된 빠른 Python 포트, 최신 readability.js와 호환되도록 업데이트됨
    - HTML 문서에서 본문 텍스트와 제목을 추출하고 정리하는 기능 제공
    - Apache License 2.0 라이선스로 배포, 다양한 버전의 Python 지원, 저자 정보 추가 및 비 ASCII HTML 처리 개선 등 업데이트

  - 🧑🏻‍💻 [NVIDIA] [NVIDIA Puts Grace Blackwell on Every Desk and at Every AI Developer’s Fingertips](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips)
    - NVIDIA가 개인용 AI 슈퍼컴퓨터인 프로젝트 DIGITS 발표
    - Grace Blackwell 플랫폼의 성능을 다양한 사람들에게 제공하는 것을 목표로 함
      - Grace Blackwell 슈퍼칩을 탑재하여 페타프롭의 AI 연산 성능 제공, 대규모 AI 모델의 프로토타이핑, 미세 조정 및 실행 지원
      - 사용자들은 데스크탑 시스템에서 모델을 개발하고 추론을 실행한 후 클라우드 또는 데이터 센터 인프라에 배포 가능

  - 🧑🏻‍💻 [Sinaptik-AI] [pandas-ai](https://github.com/Sinaptik-AI/pandas-ai)
    - PandasAI: 자연어로 데이터 질문을 쉽게 처리하는 파이썬 플랫폼
      - Jupyter Notebook, Streamlit 앱 또는 FastAPI, Flask와 같은 REST API로 배포 가능, Docker 기반 클라이언트-서버 아키텍처를 사용하여 설치 및 실행 용이
      - 데이터 프레임을 사용하여 질문하고 답변 얻는 방법 제공

  - 🧑🏻‍💻 [토니의 일기장님 tistory] [AI Agent 구축 - n8n 활용](https://starknotes.tistory.com/161)
    - n8n으로 구축한 AI Agent 워크플로우를 python으로 재현하는 방법 설명
      - (유튜브 자막 추출 → 요약 → 정보 추출 → AI agent → 데이터 병합 → 디스코드/노션 전송)
     
  - 🧑🏻‍💻 [프롬프트해커 대니님 블로그] [Claude 프롬프트 엔지니어링 완벽 가이드](https://www.magicaiprompts.com/blog/2024/04/21/claude-prompt-engineering-complete-guide)
    - Claude AI 모델의 잠재력을 극대화하기 위한 프롬프트 엔지니어링 기법을 설명해놓은 가이드
    - 명확하고 구체적인 프롬프트 작성, 다양한 예시 활용, 역할 부여 등 여러 전략을 제시하여 효과적인 프롬프트 작성법 제시
      - 명확하고 직접적인 지시, 다양한 예시 활용, 역할 부여, XML 태그 사용, 프롬프트 체이닝, 단계별 사고 유도, 응답 사전 입력, 리라이팅 요청 등

  - 🧑🏻‍💻 [Mistral AI] [Codestral 25.01](https://mistral.ai/news/codestral-2501/)
    - Mistral AI → 코드 생성 모델 Codestral 25.01을 출시
      - 기존 모델보다 코드 생성 및 완성 속도 2배 향상
      - 여러 벤치마크에서 최고 성능을 기록, 특히 FIM(fill-in-the-middle) 작업에서 탁월한 성능을 보여줌
      - Continue.dev, VS Code, JetBrains 플러그인 및 Google Cloud, Azure AI Foundry, Amazon Bedrock 등 다양한 IDE 및 플랫폼에서 사용 가능
  - 🧑🏻‍💻 [nlpai-lab] [KULLM](https://github.com/nlpai-lab/KULLM)
    - KULLM: 고려대학교 NLP & AI 연구실과 HIAI 연구소에서 개발한 한국어 특화 LLM
      - 최신 버전인 KULLM3을 포함하여 다양한 모델과 데이터셋 제공
    - KULLM3 → upstage/SOLAR-10.7B-v1.0 기반의 instruction-tuning 모델
      - 8개의 A100 GPU를 사용하여 학습됨
    - 🧑🏻‍💻 [HuggingFace] [taeminlee/KULLM3-awq](https://huggingface.co/taeminlee/KULLM3-awq)

  - 🧑🏻‍💻 [HuggingFace] [microsoft/phi-4](https://huggingface.co/microsoft/phi-4)
    - phi-4: Microsoft에서 개발한 140억 파라미터의 LLM
      - 합성 데이터셋, 필터링된 공개 도메인 웹사이트 데이터, 학술 서적 및 Q&A 데이터셋을 결합하여 훈련, 고품질 데이터와 고급 추론에 중점
      - Supervised Fine-Tuning(SFT) 및 Direct Preference Optimization(DPO)을 통해 정확한 지시 사항 준수 및 강력한 안전 조치 보장
      - MMLU, GPQA, MGSM 등 여러 벤치마크에서 우수한 성능
      - 챗 형식 프롬프트에 최적화, transformers 라이브러리를 사용하여 이용 가능
      - 영어 이외 언어의 성능 저하, 오류 발생 가능성, 고위험 시나리오 사용 시 추가적인 안전 조치 필요
    - 📜 [Microsoft Research] [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)

  - 🧑🏻‍💻 [HarderThenHarder] [RLLoggingBoard](https://github.com/HarderThenHarder/RLLoggingBoard)
    - RLHF(Reinforcement Learning from Human Feedback) 훈련 과정을 시각화하여 훈련 과정 이해 및 디버깅을 용이하게 함
    - 토큰 확률 변화, 보상 분포 변화 등을 시각적으로 보여줌
    - 주요 시각화 모듈 → 보상 영역(곡선 및 분포), 응답 영역(다양한 지표 기준 정렬), 토큰 영역(KL, 임계값, 밀집 보상, 확률)으로 구성
    - 각 영역별 지표를 통해 훈련 문제점을 파악하고 해결하는 데 도움

  - 🧑🏻‍💻 [어아인] [Titans: 차세대 메모리 아키텍처의 탄생](https://news.kojunseo.link/newsletter/14a3fbb0-4e07-4bb9-9c4f-11545fc7b6da)
    - Titans: 트랜스포머의 한계를 극복하기 위해 설계된 새로운 신경망 아키텍처, 단기 및 장기 메모리를 결합해 긴 문맥 처리와 효율성을 향상시킴
    - 장기 메모리 모듈(중요 데이터 선택/기억), 하이브리드 메모리 구조(Core, Long-term, Persistent), 최대 200만 토큰 처리 가능
    - 기존 모델 대비 낮은 perplexity, 뛰어난 긴 문맥 정보 검색 성능, 타임 시계열 및 유전체 데이터 분야에서 우수한 확장성
</details>
<details>
  <summary>4th week</summary>

  - 🧑🏻‍💻 [the decoder] [Large Language Models and the Lost Middle Phenomenon](https://the-decoder.com/large-language-models-and-the-lost-middle-phenomenon/)
    - 스탠포드, UC 버클리, 사마야 AI 연구진 → LLM이 입력 정보의 처음과 끝에 있는 정보를 가장 잘 처리한다는 것을 발견
      - 사람의 '최근 효과/초두 효과'와 유사
    - 중간에 있는 관련 정보는 성능이 현저히 저하되며, 특히 여러 문서에서 정보를 추출해야 하는 경우 더욱 심각해짐
    - LLM의 언어 처리 방식에 대한 이해와 프롬프트 디자인 개선을 통해 AI 시스템의 정보 추출 능력을 향상시킬 수 있다고 제안

  - 🧑🏻‍💻 [researchtrend] [Stay Updated on the Trends, Connect with AI Researchers](https://researchtrend.ai/)
    - AI 연구 동향을 제공하고 연구자들 간의 연결을 돕는 플랫폼
    - AI 관련 논문(arXiv)과 커뮤니티, 소셜 이벤트 정보 제공, 가격 정책 등 확인 가능
</details>

---
<details>
  <summary>My study archive 2024</summary>

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
      
    - **precision(정밀도)**: 숫자를 얼마나 정확하게 표현하는지의 정도
    - **scaling laws(스케일링 법칙)**: 모델의 크기와 성능 관계를 설명하는 규칙
    - **quantization(양자화)**: 데이터를 더 작은 비트로 압축하는 과정
    </details>
  
- 🧑🏻‍💻 [chanmuzi님 tistory](https://chanmuzi.tistory.com/479)
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
      
      - **CheckEval**: 평가의 명확성과 일관성을 높이기 위해 설계된 LLM 기반 평가 프레임워크
      - **Inter-Annotator Agreement (IAA)**: 평가자 간의 일치도를 측정하는 지표
      - **SummEval** : 요약에 대한 다양한 평가 방법을 비교하는 벤치마크 데이터셋
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
  
      - **Vision-Language Model (VLM)**: 이미지를 처리하는 비전 모델과 텍스트를 이해하는 언어 모델을 결합한 AI 모델
      - **전이 학습(Transfer Learning)**: 이미 학습된 모델을 새로운 작업에 적응시키는 방법
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
  - [2023년 최신판 OCR 8가지 API 비교평가 테스트](https://devocean.sk.com/blog/techBoardDetail.do?ID=165524&boardType=techBlog)
    - 다양한 OCR 서비스의 성능 및 속도를 비교 분석한 결과, Google Cloud Vision, Azure Document Intelligence, Upstage, Naver Clova 순으로 우수한 속도를 보임

- 🤪 [DVC] [DVC](https://dvc.org/)
  - DVC(Data Version Control): GitOps 원칙에 기반하여 대규모 데이터의 버전 관리 및 ML 모델링 프로세스의 재현 가능한 워크플로우 구축을 지원하는 오픈소스 플랫폼
  - [Github](https://github.com/iterative/dvc)

- 🧑🏻‍💻 [HuggingFace] [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
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

- 🧑🏻‍💻 [Upstage] [Solar Cookbook](https://github.com/UpstageAI/solar-prompt-cookbook)
  - Solar에 대한 프롬프트 A-Z를 담아, 누구나 더 쉽게 다룰 수 있도록 돕는 Cookbook
    - 프롬프트 엔지니어링의 기본 구조부터 복잡한 프롬프트 체이닝, 환각 해결법까지..
  - Small-Scale Model에 대한 Prompt Engineering의 insight 공유

  - [Solar_Prompt_Guide](https://github.com/studydev/Solar_Prompt_Guide)
    - Upstage Cookbook의 Prompt를 빠르게 실습할 수 있게 만들어놓은 환경
    - Upstage github repo를 fork 하여, GitHub의 CodeSpace 기반(가상 개발 컨테이너 환경)으로 필요한 몇 가지 환경 변수를 추가한 repo를 만들어놓음

- 📜 [Arcee, Florida, USA] [Arcee’s MergeKit: A Toolkit for Merging Large Language Models](https://arxiv.org/abs/2403.13257)
  <details>
      <summary>문제 상황</summary>
    
    - 특정 작업을 위해 사전 학습된 모델을 미세 조정하는 전이 학습의 발전으로 인해 수많은 작업별 특화 모델이 개발되었지만, 이들은 일반적으로 개별 작업에만 특화되어 있어 서로의 강점을 활용하지 못함
  </details>

  - MergeKit: 다수의 오픈소스 언어 모델을 효율적으로 통합하는 오픈소스 도구
    - 추가 학습 없이 모델의 성능과 다양성을 향상시키는 모델 병합 전략 지원
    - 다양한 하드웨어에서 사용 가능한 확장성 있는 프레임워크 제공
    - 이미 오픈소스 커뮤니티에서 수천 개의 모델 병합에 활용되어 Open LLM Leaderboard 상위권의 강력한 모델들을 생성하는 데 기여

  - [Github](https://github.com/arcee-ai/mergekit)

- 🧑🏻‍💻 [Amazon] [Amazon Nova and our commitment to responsible AI](https://www.amazon.science/blog/amazon-nova-and-our-commitment-to-responsible-ai)
  - Amazon Nova → Amazon에서 만든 책임감 있는 AI 개발을 위해 8가지 핵심 원칙(개인정보 보호 및 보안, 안전, 공정성, 정확성 및 견고성, 설명 가능성, 제어 가능성, 거버넌스, 투명성 등)을 바탕으로 한 새로운 멀티 모달 기반 모델
  - 이들을 제어하기 위해 SFT와 RLHF을 모두 사용하여 모델을 정렬
    - SFT → 여러 언어로 단일 및 다중 턴 훈련 데모
    - RLHF → 이전 평가의 예를 포함하여 인간의 선호도 데이터를 수집

  - 모델 개발 전 과정에서 자동화된 방법과 인간 피드백을 활용하여 편향성 평가 및 완화, 정확성 및 견고성 향상을 위한 다양한 테스트 및 벤치마크 진행, 적대적 공격에 대한 방어 및 워터마킹 기술 적용
  - 📜 [Amazon Nova Family 기술 보고서] [The Amazon Nova Family of Models: Technical Report and Model Card](https://assets.amazon.science/b0/2b/e74dd4f84f188701fd06792670e7/the-amazon-nova-family-of-models-technical-report-and-model-card.pdf)

- 🧑🏻‍💻 [Google] [python-genai](https://github.com/googleapis/python-genai)
  - Google Gen AI Python SDK: Google의 생성형 모델을 Python 애플리케이션에 통합할 수 있는 인터페이스 제공
  - 현재는 초기 출시 단계! API가 변경될 수 있으므로 프로덕션 환경에서는 사용하지 않는 것이 좋음
  - 텍스트 생성, 이미지 생성, 임베딩 등 다양한 기능 제공 및 비동기 처리 및 토큰 계산 기능 지원

- 📜 [NCSOFT] [VARCO-VISION: Expanding Frontiers in Korean Vision-Language Models](https://arxiv.org/pdf/2411.19103)
  - VARCO-VISION: 한국어와 영어를 모두 다룰 수 있는 이미지-텍스트 작업을 위해 설계된 오픈소스 VLM
    - 기존 모델의 지식을 유지하면서 시각적 정보와 언어 정보를 효과적으로 통합할 수 있도록 새로운 단계별 학습 전략 채택
  - 📊 5개의 한국어 평가 데이터셋 공개 → 4개의 폐쇄형 벤치마크, 1개의 개방형 벤치마크

    <details>
        <summary>주요 성과</summary>

      - 유사 크기의 모델과 비교해 이중언어 이미지-텍스트 이해 및 생성 능력에서 뛰어난 성능 입증
      - 다양한 기능 지원
        - Grounding: 이미지 내 객체 인식 및 위치 추적
        - Referring: 특정 객체를 지칭하는 작업
        - OCR: 이미지에서 텍스트를 추출하는 작업
    </details>

  - 🧑🏻‍💻 [HuggingFace][NCSOFT/VARCO-VISION-14B-HF](https://huggingface.co/NCSOFT/VARCO-VISION-14B-HF)

- 🧑🏻‍💻 [Google] [The next chapter of the Gemini era for developers](https://developers.googleblog.com/en/the-next-chapter-of-the-gemini-era-for-developers/)
  - Gemini 2.0 Flash: 개발자의 workflow를 개선하는 코딩 에이전트와 몰입적이고 대화형 애플리케이션 제작을 지원하는 AI플랫폼
    - 멀티모달 출력: 텍스트, 오디오, 이미지 통합 생성
    - 실시간 스트리밍 API: 오디오, 비디오 입력 지원
    - 도구 통합: Google 검색, 코드 실행 기능 지원 및 외부 도구와 연동 가능
    - AI 코드 에이전트: Jules로 자동화된 버그 수정 및 코드 작성
  - 현재는 실험 단계로 Gemini API를 통해 Google AI Studio 및 Vertex AI에서 사용 가능(내년 초 정식 출시)

- 🧑🏻‍💻 [NVIDIA] [LLaMA-Mesh:Unifying 3D Mesh Generation with Language Models](https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/?linkId=100000318302360)
  - LLaMA-Mesh: 텍스트를 기반으로 사전 학습된 LLM의 기능을 확장하여 3D Mesh를 생성할 수 있는 통합 모델

    <details>
        <summary>장점</summary>
      
      - 튜토리얼 같은 텍스트 소스에서 파생된 LLM에 내재된 공간적 지식 활용 가능
      - 대화형 3D 생성 및 Mesh 이해 가능
    </details>
    <details>
        <summary>SFT 데이터셋 구성</summary>

      - 텍스트 프롬프트로 3D Mesh 생성
      - 텍스트와 3D Mesh를 혼합한 출력 생성
      - 3D Mesh를 이해하고 해석
    </details>

  - 📜 [Tsinghua Univ., NVIDIA] [LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models](https://arxiv.org/abs/2411.09595)
</details>

<details>
  <summary>3rd week</summary>

- 📜 [FAIR at Meta, 2UC San Diego] [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
  - LLM → 언어 공간이 항상 최적의 추론 방식을 제공하지는 않음
  - Coconut(Chain of Continuous Thought): 자연어 대신 제약 없는 잠재 공간에서 LLM 추론의 가능성을 탐구하기 위해 제시한 새로운 패러다임
    - 마지막 은닉 상태를 단어로 디코딩하지 않고, 다음 입력 임베딩으로 직접 활용해 추론 효율을 높임
    - 연속적 사고 → 단일 경로에 의존X, 여러 대안의 다음 추론 단계를 인코딩해 BFS 기반 문제 해결 가능
   
- 📜 [Maryland Univ., OpenAI] [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/pdf/2406.06608)
  - 프롬프트 → GenAI 시스템과의 상호작용을 위한 주요 도구, 연구 초기 단계로 인해 용어와 개념이 혼재되어 있음
  - 목적: 프롬프트 기술의 분류 체계 구축, 주요 용어 정리 및 사용 사례 분석
    <details>
        <summary>성과</summary>

      - 어휘: 33개의 주요 프롬프트 관련 용어 정의
      - 텍스트 전용 프롬프트 기술의 분류 체계: 58가지
      - 다른 양식의 프롬프트 기술: 40가지
      - 자연어 prefix-prompting 관련 메타 분석 제시
    </details>

- 🗞️ [Google] [구글, 텍스트 프롬프트 없이 이미지 생성하는 '위스크' 공개](https://www.aitimes.com/news/articleView.html?idxno=166297)
  - 위스크(Whisk) → Google이 공개한 이미지 생성 AI
    <details>
          <summary>작동 방식</summary>

      - 구글의 이미지 생성 모델 Imagen 3 기반
      - 텍스트 프롬프트 대신 3가지 이미지(주제 이미지, 장면 이미지, 스타일 이미지)를 결합하여 새로운 이미지 생성
      - 입력 이미지를 바탕으로 자동 생성된 텍스트 캡션을 활용해 이미지 생성
    </details>
  - [Whisk](https://labs.google/fx/tools/whisk/unsupported-country)

- 🗞️ [Google] [Veo 2](https://deepmind.google/technologies/veo/veo-2/)
   - Veo 2: Google DeepMind에서 개발한 최첨단 비디오 생성 모델
   - 메타의 MovieGenBench 데이터셋 기반
   - 🗞️ ["구글의 비오 2, 소라에 압승"...테스터 비교 영상 속속 등장](https://www.aitimes.com/news/articleView.html?idxno=166379)

- 📜 [NYU] [Self-Reflection Outcome is Sensitive to Prompt Construction](https://arxiv.org/abs/2406.10400)
    - LLMs → zero-shot 및 few-shot 추론 능력을 보여줌 → Self-Reflection으로 개선 가능함을 제안
      - LLM 스스로 초기 응답의 실수를 식별하고 수정하게끔

      <details>
          <summary>주요 발견</summary>
        
        - 기존 Self-reflection 연구에서 사용된 대부분의 프롬프트는 편향을 포함 → LLM이 정답을 불필요하게 수정하도록 유도
        - 보수적인 프롬프트 설계를 통해 Self-Reflection의 정확도 향상을 입증
      </details>
  - [Github](https://github.com/Michael98Liu/mixture-of-prompts)
 
- 🧑🏻‍💻 [LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct)
  - EXAONE 3.5 → LG AI Research에서 개발한 이중언어(영어, 한국어) 생성 모델로 장문 처리 기능 지원(32K 토큰까지)
  - TensorRT-LLM, vLLM 등 여러 추론 프레임워크 지원 → 다양한 환경에서 배포 및 활용 가능
    <details>
        <summary>한계</summary>
   
      - 편향된 반응을 보일 수 있음
      - 최신 정보를 반영하지 않아 응답이 거짓/모순될 수 있음
      - 의미적으로 잘못된 문장이 생성될 수 있음
    </details>
  - [Github](https://github.com/LG-AI-EXAONE/EXAONE-3.5), [Blog](https://www.lgresearch.ai/blog/view?seq=507)
  - 📜 [LG AI Research] [EXAONE 3.5:Series of Large Language Models for Real-world Use Cases](https://arxiv.org/pdf/2412.04862)
 
- 🧑🏻‍💻 [BE_성하님 tistory] [DB Lock이란?(feat. Lock 종류, 블로킹, 데드락)](https://ksh-coding.tistory.com/121)
  - DB Lock: 동시에 여러 트랜잭션이 데이터를 변경하는 것을 방지하여 데이터 무결성을 유지하는 메커니즘
  - 공유 락(S Lock)과 배타 락(X Lock)이 있으며, 사용에 따라 Blocking 현상이나 Deadlock이 발생할 수 있음
    - Blocking은 한 transaction이 다른 transaction이 lock을 해제할 때까지 기다리는 현상
    - Deadlock은 두 개 이상의 transaction이 서로 상대방의 lock을 기다리며 영원히 진행되지 않는 상황
</details>

<details>
  <summary>4th week</summary>

- 🧑🏻‍💻 [OpenAI] [OpenAI o3 Breakthrough High Score on ARC-AGI-Pub](https://www.youtube.com/watch?v=SKBG1sqdyIU&ab_channel=OpenAI](https://arcprize.org/blog/oai-o3-pub-breakthrough))
  - OpenAI의 o3가 ARC-AGI-Pub의 Semi-Private 평가 세트에서 75.7%라는 점수를 기록 (고성능 설정 → 87.5%)
  - 기존 LLM의 한계인 테스트 시간에 지식을 재결합하는 능력 부족을 극복하여 자연어 프로그램 탐색을 통해 새로운 수준의 적응력과 일반화 능력을 보여줌
    <details>
        <summary>중요 개념</summary>
   
      - **ARC-AGI**: AI의 일반적인 추론 능력을 평가하기 위해 만들어진 데이터셋
      - AGI(Artificial General Intelligence)
    </details>

- 🧑🏻‍💻 [HuggingFace] [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)
  - Granite-3.1-8B-Instruct: 복잡한 장문 컨텍스트를 해결할 수 있도록 설계된 모델로, Granite-3.1-8B-Base에서 지도 학습, 강화 학습 기반 정렬, 모델 병합 등의 기술을 활용해 파인튜닝됨
    <details>
        <summary>적용 분야</summary>
      
      - 요약, 텍스트 분류/추출, 질의응답,RAG(검색 기반 생성), 코드 작업, 함수 호출,다국어 대화, 장문 컨텍스트 작업 등
      - 모델 구조: 디코더 전용 dense 트랜스포머 (RoPE, SwiGLU, RMSNorm 등 포함)  
      - 시퀀스 길이: 최대 128K 토큰  
    </details>
    <details>
        <summary>중요 개념</summary>
      
      - **SFT(Supervised Fine-Tuning)**: 사전 훈련된 언어 모델을 특정 작업이나 도메인에 맞게 조정하는 과정(지도 학습 기반 미세 조정)
      - **dense**: 인공지능 모델의 아키텍처에서 매개변수가 고르게 분포되고 사용되는 구조
        - 🧐 사용 이유: 모델의 일관성 있는 성능, 다양한 작업에서의 일반화를 보장하기 위함
      - **dense transformer**: 각 모델 레이어가 동일한 매개변수로 구성되고, 모든 뉴런과 연결이 활성화된 상태에서 작동하는 전통적인 트랜스포머 아키텍처
        - 반대 개념: MoE(Mixture of Experts) 아키텍처는 일부 뉴런만 활성화
    </details>
  - [Github](https://github.com/ibm-granite/granite-3.1-language-models)

- 🧑🏻‍💻 [HuggingFace] [nlpai-lab/KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1)
  - KURE-v1: BAAI/bge-m3 모델을 한국어 데이터로 미세 조정한 것으로, 공개된 한국어 검색 모델 중 최고 성능을 보임
    - 1024차원, 최대 8192 토큰 길이 지원
    - Recall, Precision, NDCG, F1 등의 지표에서 우수한 성능 기록
    - 파인튜닝 방식: 다양한 한국어 문서 검색 데이터셋을 사용하여 학습 (CachedGISTEmbedLoss 활용)
  - [Github](https://github.com/nlpai-lab/KURE)
    <details>
        <summary>중요 개념</summary>
      
      - **Recall(재현율)**: 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율
        - 실제 관련 문서 중에서 검색 모델이 얼마나 많이 찾아냈는지
        - Recall = (검색된 관련 문서 수) / (전체 관련 문서 수)
        - Recall 값이 높음 → 관련된 문서를 빠뜨리지 않고 잘 찾아낸다​
      - **Precision(정밀도)**: 여러 번 측정하거나 계산하여 그 결과가 서로 얼만큼 가까운지를 나타내는 기준
        - 검색된 문서 중에서 실제로 관련 있는 문서의 비율
        - Precision= (검색된 관련 문서 수) / (검색된 전체 문서 수)
        - Precision 값이 높음 → 검색된 문서들이 대부분 관련성이 있다
      - **NDCG(Normalized Discounted Cumulative Gain)**: 모델이 예측한 순위를 반영한 측정 지표
        - 검색 결과의 순서를 고려하여, 상위에 있는 검색 결과가 얼마나 관련성이 높은지 평가
        - NDCG= (DCG) / (IDCG)
          - DCG: 검색 결과 순서에 따라 가중치를 부여한 누적 점수
          - IDCG: 최적의 순서에서 얻을 수 있는 최대 DCG 값
        - NDCG 값이 높음 → 관련 문서가 상위에 많이 배치된다
      - **F1**: Recall과 Precision의 조화 평균 (두 지표 간의 균형)
        - F1= 2 × {(Precision x Recall)/(Precision + Recall)}
        - F1 값이 높음 → Recall과 Precision 둘 다 우수하다
    </details>

- 🧑🏻‍💻 [Cosmograph](https://cosmograph.app/docs/cosmograph/Cosmograph%20Python/get-started-widget/)
  - Cosmograph: 그래프 데이터셋 & 벡터 임베딩 시각화 프레임워크 (복잡한 데이터 관계를 시각화하여 데이터 분석 기능을 향상시킴)
  - Anywidget 기반으로 Jupyter 환경과 원활하게 통합되어 대화형 그래프 제공

- 🧑🏻‍💻 [HuggingFace] [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)
  - ModernBERT: 2조 토큰의 영어 및 코드 데이터로 사전 훈련된, 최대 8,192 토큰의 긴 문맥을 처리할 수 있는 현대화된 양방향 인코더 전용 Transformer 모델
    <details>
        <summary>주요 특징</summary>

      - RoPE (Rotary Positional Embeddings): 긴 문맥 지원 강화
      - Local-Global Alternating Attention: 긴 입력 처리 효율성 향상
      - Unpadding 및 Flash Attention: 빠른 추론 지원
    </details>
    <details>
        <summary>활용 분야</summary>

      - 긴 문서 처리: 검색, 분류, 대규모 코퍼스 내 의미 검색
      - 코드 검색: 코드 검색 및 텍스트 + 코드 혼합 의미 검색
    </details>

- 📜 [The Super Weight in Large Language Models](https://arxiv.org/abs/2411.07191)
  - LLM에서 일부 극단값(outliers) 파라미터가 모델 성능에 매우 중요한 영향을 미친다는 것을 발견
    - 단 하나의 파라미터 제거로도 perplexity가 1000배 증가하고, zero-shot 정확도가 추측 수준으로 하락
    - Super Weights: 단일 forward pass로 데이터 없이 중요한 파라미터를 식별하는 방법
    - Super Activations: 큰 활성화를 유발하는 드문 파라미터, 이를 보존하면 모델 성능이 크게 향상됨
    <details>
        <summary>Weight Quantization 개선</summary>

      - Super weights를 보존하고 다른 outliers를 클리핑하여, 단순한 round-to-nearest quantization로 최첨단 성능 달성 가능
      - 기존보다 더 큰 블록 크기에서도 효과적인 양자화 구현 가능 (양자화 기술의 한계 확장)
    </details>
    <details>
        <summary>중요 개념</summary>
      
      - **극단값 (Outliers)**: 다른 값들과 큰 차이를 보이는 데이터 포인트
      - **Perplexity**: 언어 모델의 예측 품질을 측정하는 지표로, 낮을수록 예측 정확도가 높음을 의미
      - **Zero-shot**: 학습되지 않은 작업에 대해 모델이 직접 일반화하여 수행하는 능력을 측정하는 평가 방식
      - **Forward pass**: 모델이 입력 데이터를 통해 예측을 생성하는 과정, 파라미터의 활성화 값을 계산
      - **Weight quantization**: 모델의 가중치를 정밀도를 낮춘 형식으로 표현해 메모리와 계산 자원을 절감하는 기술
      - **Super weights**: 모델 성능에 결정적인 영향을 미치는 중요한 가중치 파라미터
      - **Round-to-nearest quantization**: 가장 가까운 정밀도 수준으로 값을 반올림하는 간단한 양자화 방법
        - **양자화(Quantization)**: 모델의 가중치나 활성화를 낮은 비트 정밀도로 변환하여 메모리 사용량과 계산 비용을 줄이는 기법, 모델의 성능 손실을 최소화하면서 경량화 및 최적화를 목표로 함
    </details>

- 🧑🏻‍💻 [Philschmid] [How to fine-tune open LLMs in 2025 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2025?utm_source=substack&utm_medium=email)
  - 2025년 기준 Hugging Face를 활용한 오픈 LLM 파인튜닝 방법을 설명해놓은 사이트
    - QLoRA, Spectrum 등 최적화 기법과 분산 학습을 중점적으로 다룸
    - 파인튜닝 전에 프롬프트 엔지니어링이나 기존 파인튜닝된 모델 활용 가능성을 평가하고, 효율적인 파인튜닝을 위해 QLoRA 또는 Spectrum기법을 활용할 것을 제안
    - 다양한 하드웨어 및 DeepSpeed를 이용한 다중 GPU 분산 학습 환경 설정과 Flash Attention 및 Liger Kernels 등 최적화 전략을 통해 학습 시간을 단축하는 방법 제시
    <details>
        <summary>중요 개념</summary>

      - **분산 학습 (Distributed Training)**: 모델 학습을 여러 GPU 또는 노드로 분산하여 처리 속도를 높이고, 대규모 데이터와 모델을 효율적으로 처리하는 학습 방법.
      - **Fine-tuning**: 이미 학습된 모델을 특정 작업이나 데이터셋에 맞게 추가로 학습시켜 성능을 개선하는 과정
      - **QLoRA (Quantized LoRA)**: 양자화된 모델에서 저렴한 학습 가능한 적응 계층(LoRA)을 활용하여 고성능 파인튜닝을 가능하게 하는 기법, 메모리와 계산 비용을 크게 절감
      - **Spectrum**: 모델 학습 중 다양한 대역폭과 데이터 표현 방식을 최적화해 학습 효율성을 높이는 기법, 특히 분산 학습에서 자원 활용도 향상
      - **Flash Attention**: GPU 메모리와 연산을 효율적으로 사용하여 Transformer 모델에서 Attention 연산 속도를 크게 향상시키는 최적화 기법
      - **Liger Kernels**: 커널 수준에서 GPU 활용도를 극대화하도록 설계된 최적화 기술, 대규모 모델 학습 시 효율적인 연산 분배를 통해 학습 시간 단축
    </details>

- 🧑🏻‍💻 [LMArena] [WebDev Arena Leaderboard](https://web.lmarena.ai/leaderboard)
  - WebDev Arena: LMArena가 개발한 웹 개발 AI 성능 벤치마크
    - Claude 3.5 Sonnet이 1위, 다음으로 o1-mini, Gemini-Exp-1206 등이 상위권을 기록
    - 순위표는 Arena Score, 95% 신뢰구간, 투표 수 등을 포함하여 각 모델의 성능을 상세히 비교
    - 더 자세한 통계는 평균 승률, 모델 간 승리 비율, 대결 횟수 등의 추가 그래프를 통해 확인 가능
   
- 🧑🏻‍💻 [HuggingFace] [deepseek-ai/DeepSeek-V3-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)
  - DeepSeek-V3-Base: 685B 파라미터의 크기를 가진 강력한 MoE 언어 모델
    - 각 토큰에 대해 37B 매개변수 활성화
    - 효율적인 추론과 비용 절감을 위해 MLA, DeepSeekMoE 아키텍처 사용
    <details>
        <summary>주요 특징</summary>
      
      - 보조 손실 없이 부하 균형 유지
      - 다중 토큰 예측(MTP) 학습 목표로 성능 강화
      - FP8 혼합 정밀도 훈련을 통한 14.8조 토큰으로 사전 학습
      - 효율적인 통신 설계로 훈련 비용과 시간 최소화
      - NVIDIA, AMD GPU, Huawei Ascend NPU 등 다양한 하드웨어 지원
      - SGLang, LMDeploy, TensorRT-LLM 등으로 로컬에서 실행 가능
    </details>
  - [Github](https://github.com/deepseek-ai/DeepSeek-V3)
    <details>
        <summary>중요 개념</summary>
      
      - **MoE (Mixture of Experts)**: 각 입력 토큰에 최적의 expert를 선택해 연산 부담 감소 및 성능 극대화
      - **MLA (Multi-Level Activation)**: 계산 자원을 효율적으로 배분하고 학습 및 추론 성능을 최적화하는 기법
      - **MTP (Multi-Token Prediction)**: 모델이 한 번에 여러 토큰을 예측하도록 학습, 모델 성능을 강화
    </details>
</details>
</details>
