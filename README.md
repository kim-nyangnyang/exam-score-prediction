# exam-score-prediction
Kaggle exam score prediction project

# 📖 효율적 성적 향상을 위한 전략 수립: 머신러닝&통계분석


 **"성적에 결정적 영향을 미치는 핵심 동인은 무엇이며, 복합요인 간의 비선형 관계를 학습하여 성적을 정밀하게 예측할 수 있는 최적의 모델은 무엇인가?"**


높은 교육열로 인해 학생들의 성적은 큰 관심의 대상이며 학생들은 효과적 성적 향상을 위해 다양한 노력을 한다.
하지만 현대 교육환경에서 성적은 **다양햔 요인들과 복잡한 상호작용을하고 비선형적 관계**를 가지고 있다. 
특정 요인이 성적에 미치는 진정한 기여도를 산출하고 정확한 예측을 하기 위해서는 이러한 관계를 고려해야 한다.


## 1. Project Overview 
- **주제** : 성적에 영향을 주는 핵심 요인을 도출, 성적 예측 모델링을 통한 효율적인 학습처방 제공
- **데이터셋** : [Predicting Student Test Scores](https://www.kaggle.com/competitions/playground-series-s6e1/data)
- **핵심 목표** : 
1. **예측모델 구축**: 비선형 패턴과 변수 간 상호작용을 자동 학습하는 트리 기반 부스팅 모델을 적용하여 예측 성능을 극대화한다.
2. **성적 결정 핵심 동인 규명**: 변수 중요도 분석을 통해, 단순 상관관계를 넘어 성적 향상에 실질적으로 기여하는 핵심 동인을 추출한다.
3. **데이터 기반의 개인별 학습 처방 제공:** 분석된 핵심 요인을 바탕으로 위험군을 조기 식별하고, 학생별 성적 향상 맞춤형 시나리오를 제시한다.

🛠 Tech Stack
- Data Analysis & Stats
  - pandas
  ,numpy
  ,scipy
  ,statsmodels
  ,scikit-posthocs
- Machine Learning
  - plotly
  ,matplotlib
  ,seaborn
- Visualization
  - scikit-learn
  ,lightgbm
  ,xgboost
  ,catboost

## 2. Data Dictionary (주요 핵심 변수)
- 실제 분석 결과를 통해서 확보한 변수들의 기재
- 총 변수갯수 : 12개

## 📊 데이터셋 명세 (Dataset Specifications)

Kaggle의 시험 성적 예측 챌린지 데이터를 기반으로 하며, 변수의 특성에 따라 3개 지표로 분류함.

### 🔍 변수 정의 및 상세 설명


| 분류 | 변수명 | 설명 | 데이터 타입 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **기본 정보** | `student_id` | 학생 고유 식별 번호 | `String / ID` | 고유값 (Unique) |
| | `age` | 학생의 연령 | `Numeric` | |
| | `gender` | 성별 | `Categorical` | Male, Female, Other |
| **학업 지표** | `course` | 수강 중인 학위 과정 | `Categorical` | B.Tech, B.Sc 등 7개항목 |
| | `study_hours` | 일평균 학습 시간 | `Numeric` | hours/day |
| | `class_attendance` | 수업 출석률 | `Numeric` | 0% ~ 100% |
| | `study_method` | 주요 학습 방법 | `Categorical` | 독학, 그룹스터디, 인터넷강의 등 |
| | `exam_difficulty` | 시험 난이도 | `Ordinal` | easy, moderate, hard |
| **생활 환경** | `internet_access` | 인터넷 가용 여부 | `Binary` | Yes / No |
| | `sleep_hours` | 일평균 수면 시간 | `Numeric` | hours/day |
| | `sleep_quality` | 수면의 질 수준 | `Ordinal` | Poor, Average, Good |
| | `facility_rating` | 시설물 수준 | `Ordinal` | low, mediun, high |
| **targetVariable** | **`exam_Score`** | **시험점수 (Target)** | `Numeric` | 0 ~ 100 |


---
## 3. Problem Definition
- **데이터 특성** 
    1. **복합적 변수 구성** : 응답자의 특성을 다양한 독립변수로 나타냄
    2. **수치형과 범주형의 혼재** : 전처리 필수
    3. **비선형적 관계 가능성** : 나이, 학습시간, 수면시간 등 복합적 비선형선 존재
    
- **분석 방향**
    + 통계분석 : 효과크기, 상관분석, ANOVA, MI Score
    + 머신러닝 : XGBoost, LightGBM Catboost 앙상블

## 4. 🛠️ Data Preprocessing
- **데이터 스케일링**: 수치형 데이터에 `StandardScaler` 적용
- **범주형 변수 처리**: 순서형 변수는 `Ordinal Encoding`, 일반 범주형은 `One-Hot Encoding` 적용.

## 5. 통계분석 핵심 인사이트

### ✅ 다중공선성(VIF) 확인 결과
- **문제 진단**: 모든 변수VIF 2 이하로 다중공선성 문제 없으나 파생변수 생성 시 다중공선성 문제 발생 가능
- **해결 전략**: 모델사용 간 tree기반 모델 사용으로 다중공선성 문제 완화, 중요도 떨어지는 변수 제거

### ✅ 효과 크기(Effect Size) 분석
단순 p-value 유의성을 넘어, 실제 성적에 기여하는 정도를 **에타 제곱(η²)**으로 정량화
| 순위 | 변수명 | 효과 크기 ($\eta^2$) | 영향력 수준 |
|:---:|:---|:---:|:---:|
| 1 | **Study Hours** | **0.5811** | **Large** |
| 2 | Class Attendance | 0.1303 | Medium |
| 3 | Sleep Quality | 0.0561 | Small |
| 4 | Study Method | 0.0469 | Small |
| 5 | Facility Rating | 0.0354 | Small |

### ✅ 상관분석
<img width="800" height="500" alt="상관분석" src="https://github.com/user-attachments/assets/eff93fc5-4287-49c9-a4d6-319b2321a3ea" />

### ✅ Mutual Information Score
| 순위 | 변수명 | MI Score | 비고 |
|:---:|:---|:---:|:---|
| 1 | **Study Hours** | **0.9117** | 예측 기여도 압도적 |
| 2 | Class Attendance | 0.3019 | 주요 관리 지표 |
| 3 | Sleep Hours | 0.0731 | 유의미한 지표 |
| 4 | Study Method | 0.0422 | 방법론적 차이 발생 |

### ✅ 파생 변수의 도입
- **age_ord**: 나이에 따른 점수의 비선형성을 고려해 나이를 구간화한 ordinal 변수(19,22세 기준) 
---

### 🔍 머신러닝에 사용한 변수
xgboost를 기반으로 한 변수중요도 실험을 통해 삭제할 변수 선정(age, course, exam_difficulty, internet_access, gender )

| 분류 | 변수명 | 설명 | 데이터 타입 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **학업 지표** | `course` | 수강 중인 학위 과정 | `Categorical` | B.Tech, B.Sc 등 7개항목 |
| | `study_hours` | 일평균 학습 시간 | `Numeric` | hours/day |
| | `class_attendance` | 수업 출석률 | `Numeric` | 0% ~ 100% |
| | `study_method` | 주요 학습 방법 | `Categorical` | 독학, 그룹스터디, 인터넷강의 등 |
| **생활 환경** | `sleep_hours` | 일평균 수면 시간 | `Numeric` | hours/day |
| | `sleep_quality` | 수면의 질 수준 | `Ordinal` | Poor, Average, Good |
| | `facility_rating` | 시설물 수준 | `Ordinal` | low, mediun, high |



### 💡 선택 기준 (Feature Selection Rationale)

통계분석 결과를 기반으로, xgboost를 활용해 변수 중요도를 산정하고, 중요도가 낮은 변수를 삭제하며 RMSE 차이 확인

## 6. 모델링 평가지표

| Model |RMSE | Weight|
| :--- | :--- | :--- |
| Lightgbm| 8.849688 | 0.333 |
| XGBoost | 8.827545 | 0.3336 |
| CatBoost | 8.828255 | 0.3334 |
| ensemble | 8.6553 |  |


> **Note** : 최종 대회 public score : 8.72952 / private: 8.75411



## 7. 🔍 Feature Importance (SHAP Analysis)

SHAP 분석을 통해 예측 모델이 각 개인을 판단할 때 중요하게 고려한 지표의 순위를 도출
<img width="789" height="579" alt="sharp" src="https://github.com/user-attachments/assets/e7ac2fa7-0114-4fe4-82db-abcbb9e14db3" />
**[Top 5 핵심 인자]**
1. Study Hours (학습 시간): 모델 output에 가장 강력한 영향을 미치며, 학습 시간이 많을수록(Red) 예측 성적이 비례하여 크게 상승
2. Class Attendance (출석률): 학습 시간 다음으로 중요한 변수이며, 출석률이 높을수록(Red) 양(+)의 방향으로 모델의 성적 예측값이 이동 
3. Sleep Quality (수면의 질): 수면의 질이 우수할수록 성적 예측에 긍정적인 기여를 하는 반면, 수면 시간이 너무 적거나 질이 낮으면(Blue) 부정적인 영향 존재
4. Study Method (학습 방법): 특히 Coaching 방식이 성적 향상에 유의미한 양의 영향을 주며 단순히 평균 점수를 올리는 것을 넘어, 특정 조건에서 **성적의 도약**을 이끌어내는 핵심 변수임


> **💡 요약**: 학습량이 압도적인 영향력을 보이며, 성실성과 휴식의 질이 데이터적으로 점수에 큰 기여를 한다. 학습방법 중 특히 자습과 코칭의 점수차이가 크며, 코칭은 유의미한 성적의 도약을 이끌어낸다.




## 8. Conclusion
### 🎯 활용방안
1. **성적예측을 통한 성적 저조군 조기 식별** : 성적 저조군 식별 및 조기개입을 통한 효율적 관리
2. **성적향상 가이드라인** : 주요 요인(학습시간, 참석률 등)위주의 성적향상 가이드라인 제공
3. **개인 맞춤 전략** : 학습시간의 점진적 향상, 학습방법 변경 등의 전략 / 저소득층의 경우 방과후 수업, 국가시행 무료과외 등 방안 제시
4. **생활습관 개선** : 적절한 수면시간 확보, 수면시간에는 수면만 취하며 수면의 질 확보


## ⭐상세 내용 확인
- 프로젝트 상세 보고서는 PDF 슬라이드 자료를 참고해 주세요

- 분석코드 : [분석코드](report/randomsearch.ipynb)
* [📑 성적 예측 가이드라인 보고서 (PDF)](report/exam_score_predict.pdf)

# 🔗 배지 및 이모지 공식 소스 링크
| 용도 | 사이트 이름 | 링크 |
| :--- | :--- | :--- |
| **배지 생성** | Shields.io | [https://shields.io/](https://shields.io/) |
| **로고/색상 검색** | Simple Icons | [https://simpleicons.org/](https://simpleicons.org/) |
| **이모지 검색** | Emoji Cheat Sheet | [https://github.com/ikatyang/emoji-cheat-sheet](https://github.com/ikatyang/emoji-cheat-sheet) |
