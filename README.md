# final-project-level3-recsys-10

## ❗ 주제 설명
- 유저의 좋아하거나 싫어하는 맥주 정보를 입력 받아 편의점의 맥주 4캔을 추천해주는 서비스 

## 📅 프로젝트 수행 기간 
- 2022.05.16 ~ 2022.06.10



## 👋 팀원 소개

|                                                  [신민철](https://github.com/minchoul2)                                                   |                                                                          [유승태](https://github.com/yst3147)                                                                           |                                                 [이동석](https://github.com/dongseoklee1541)                                                  |                                                                        [이아현](https://github.com/ahyeon0508)                                                                         |                                                                         [임경태](https://github.com/gangtaro)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/52911772?v=4)](https://github.com/minchoul2) | [![Avatar](https://avatars.githubusercontent.com/u/39907669?v=4)](https://github.com/yst3147) | [![Avatar](https://avatars.githubusercontent.com/u/41297473?v=4)](https://github.com/dongseoklee1541) | [![Avatar](https://avatars.githubusercontent.com/u/44939208?v=4)](https://github.com/ahyeon0508) | [![Avatar](https://avatars.githubusercontent.com/u/45648852?v=4)](https://github.com/gangtaro) |


## 🏢 Structure
```bash
final-project-level3-recsys-10
├── 📁 EDA
│   └── ⋮
├── 💾 README.md
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
├── poetry.lock
└── pyproject.toml
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




## 📜 Reference
- [Liang, Dawen, et al. "Variational autoencoders for collaborative filtering." Proceedings of the 2018 world wide web conference. 2018.](https://dl.acm.org/doi/pdf/10.1145/3178876.3186150)

- [Kang, Wang-Cheng, and Julian McAuley. "Self-attentive sequential recommendation." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.](https://arxiv.org/pdf/1808.09781.pdf)

- [Sedhain, Suvash, et al. "Autorec: Autoencoders meet collaborative filtering." Proceedings of the 24th international conference on World Wide Web. 2015.](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)

- [beerinfo](https://beerinfo.net/)
- [ratebeer](https://www.ratebeer.com/)
