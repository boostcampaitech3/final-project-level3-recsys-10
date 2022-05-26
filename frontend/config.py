class CGF:

    # coldstart 칼럼과 아이템 개수 
    max_col = 4
    num_items = 12 # max_col의 배수로 지정해주세요

    # radio-box 에 들어갈 옵션
    options = ["좋아요😊", "몰라요🙄", "싫어요😠"]

    # 고정할 이미지 사이즈
    image_size = [150,420]

    # 좋아요 싫어요 몰라요 변형점수
    LIKE = 5
    UNLIKE = 1
    UNKNOWN = -1

    # 추천 키워드
    rec_keywords = ["새로운",
                    "청량한",
                    "치킨",
                    "구수한",
                    ]

