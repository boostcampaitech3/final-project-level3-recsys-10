# final-project-level3-recsys-10

## ❗ 프로젝트 소개
- 유저의 취향에 맞는 편의점 맥주 4캔 추천 서비스 

## 📅 프로젝트 수행 기간 
- 2022.05.16 ~ 2022.06.10

## 👋 팀원 소개

|                                                  [신민철](https://github.com/minchoul2)                                                   |                                                                          [유승태](https://github.com/yst3147)                                                                           |                                                 [이동석](https://github.com/dongseoklee1541)                                                  |                                                                        [이아현](https://github.com/ahyeon0508)                                                                         |                                                                         [임경태](https://github.com/gangtaro)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/52911772?v=4)](https://github.com/minchoul2) | [![Avatar](https://avatars.githubusercontent.com/u/39907669?v=4)](https://github.com/yst3147) | [![Avatar](https://avatars.githubusercontent.com/u/41297473?v=4)](https://github.com/dongseoklee1541) | [![Avatar](https://avatars.githubusercontent.com/u/44939208?v=4)](https://github.com/ahyeon0508) | [![Avatar](https://avatars.githubusercontent.com/u/45648852?v=4)](https://github.com/gangtaro) |

## 📁 데이터 개요
- [Ratebeer](https://www.ratebeer.com/) 해외 맥주 리뷰 커뮤니티 크롤링
- 유저의 다양한 맥주의 선호도를 수집할 수 있음 → user-item interaction
- 한국 내에 유통 되는 거의 모든 세계 맥주에 대한 데이터가 존재
- Explicit dataset (맥주의 선호도로 판단할 수 있는 변수 존재)
- sparsity가 96.29(%)로 High Sparsity 데이터

![image](https://user-images.githubusercontent.com/44939208/173780828-753fa866-0561-47c3-9551-fce484f7414a.png)

## 🔧 데이터 파이프라인
![image](https://user-images.githubusercontent.com/44939208/173781375-71b6dbaa-5f04-4c03-a8ec-e09b542855e2.png)

## 💎 시스템 아키텍처
<img width="850" alt="system architecture" src="https://user-images.githubusercontent.com/44939208/173782823-21325b15-8934-4347-883c-9c34ca266599.png">

## 🎓 Model
- 모델 선정 기준
    - 개인화 추천 가능
        - 유저의 취향을 파악하여 추천해주는 서비스이기 때문에, 룰베이스 모델의 개발은 맞지 않음
    - User Free 모델 (유저 파라미터를 학습하지 않는 모델)
        - 학습된 유저에게 추천을 해주는 것이 아닌, 새로운 유저의 정보로 추론이 가능해야함
    - Sparse data
        - 희소도가 매우 높은 데이터셋
        - 해당 데이터셋에 대응 가능한 모델링 방법론이 적용되어 있어야 함  
        → Feature Embedding, Latent vector, …  
        
        
<h3>=>  AutoRec</h3>

<img width="850" alt="autorec" src="https://user-images.githubusercontent.com/44939208/173783005-7842da1f-abbe-4ba3-a524-ee1b7630c3be.png">

## 🏢 Structure
```bash
final-project-level3-recsys-10
├── 📁 EDA
│   └── ⋮
├── 💾 README.md
├── 📁 .github
│   ├── 📁 ISSUE_TEMPLATE
│   │    └── 💾 Issue-template.md
│   ├── 📁 workflows
│   │    └── 💾 docker-publish.yml
│   └── 💾 PULL_REQUEST_TEMPLATE.md
├── 📁 backend
│   ├── 📁 app
│   │    ├── 📁 DB
│   │    │    ├── 💾 crud.py
│   │    │    ├── 💾 database.py
│   │    │    ├── 💾 models.py
│   │    │    ├── 💾 schemas.py
│   │    │    └──  ⋮
│   │    ├── 💾 __main__.py
│   │    ├── 💾 main.py
│   │    └── 📁 routers
│   │        └──  ⋮
│   └── 📁 recommendAPI
│       ├── ⋮
│       └── 📁 s3rec
│           └── ⋮
├── 📁 data_engineering
│   └── ⋮
├── 📁 frontend
│   ├── 📁 static
│   │   ├── 📁 css
│   │   │   └── ⋮
│   │   ├── 📁 img
│   │   │   └── ⋮
│   │   └── 📁 js
│   │       └── ⋮
│   ├── 📁 templates
│   │   └── ⋮
├── 📁 model
│   └── ⋮
├── 💾 .gitignore
├── 💾 Dockerfile
├── 💾 Makefile
├── 💾 start.sh
└── 💾 requirements.txt
```

## 🏃‍ Run
```
pip install -r requirements.txt
python -m backend.app
```

## 🎞 Demo
- 서빙을 위한 프론트 페이지
<img width="500" height="300" alt="Front Page" src="https://user-images.githubusercontent.com/41297473/172408055-1774782b-848f-435d-bd93-048ae9a0668e.gif">

- 유저의 Cold start를 해결하기 위한 페이지
<img width="500" height="300" alt="Cold start" src="https://user-images.githubusercontent.com/41297473/172411181-f71e3d52-edf7-485d-a070-dd9764475c12.gif">

## 👨‍👩‍👧‍👧 Collaborate Working
- Github Issues 기반 작업 진행
<img width="500" height="300" alt="Git Issues" src="https://user-images.githubusercontent.com/41297473/172408276-b164089a-6f57-4ad3-ad4f-d0772bdf08bb.gif">

- Github Projects, Notion의 칸반 보드를 통한 일정 관리
<img width="500" height="300" alt="Git Projects" src="https://user-images.githubusercontent.com/41297473/172408451-0bae191b-043a-4130-b61b-3790fdf79117.gif">
<br><img width="500" height="300" alt="notion_kanban" src="https://user-images.githubusercontent.com/41297473/172409727-ec1e817c-6711-42ef-8b18-32c34bb5574d.gif"></br>

- Github Pull Request 를 활용한 브랜치 관리
<img width="500" height="300" alt="Git Pull Request" src="https://user-images.githubusercontent.com/41297473/172410317-d2697b7e-4889-4672-a064-b653095d17aa.gif">

## ✨ WRAP-UP REPORT
- [WRAP-UP REPORT](https://poised-speedwell-186.notion.site/Final-WRAP-UP-REPORT-f32fcafa56ce4ce78132058aed8a56aa)

## 📜 Reference
- [Liang, Dawen, et al. "Variational autoencoders for collaborative filtering." Proceedings of the 2018 world wide web conference. 2018.](https://dl.acm.org/doi/pdf/10.1145/3178876.3186150)
- [Kang, Wang-Cheng, and Julian McAuley. "Self-attentive sequential recommendation." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.](https://arxiv.org/pdf/1808.09781.pdf)
- [Sedhain, Suvash, et al. "Autorec: Autoencoders meet collaborative filtering." Proceedings of the 24th international conference on World Wide Web. 2015.](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)
- [beerinfo](https://beerinfo.net/)
- [ratebeer](https://www.ratebeer.com/)
