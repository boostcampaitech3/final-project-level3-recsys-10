from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, Field
# from uuid import UUID, uuid4
from datetime import datetime

class User(BaseModel):
    """
    description: 유저 한 명에 대한 정보
    """
    userID: int
    profileName: str


class UserCreate(User):
    """
    despription: 새로운 유저 추가
    application: 회원가입
    """
    password: int
    gender: str
    birth: datetime = Field(default_factory=datetime.now)

class Beer(BaseModel):
    """
    description: 맥주 하나에 대한 정보
    """
    # 맥주 하나에 대한 정보
    beerName: str
    beerID: int
    brewerID: int
    ABV: float
    style: str
    imageUrl: str


# 지금은 사용하지 않는 기능
# class BeerCreate(Beer):
#     """
#     despription: 새로운 맥주 추가
#     application: 신제품 등록
#     """
#     launchedTime: datetime = Field(default_factory=datetime.now)

class BeerReview(BaseModel):
    """
    description: 하나의 맥주 리뷰. # 지금 현재, 사용하는 인자만 활성화 시켜놓았음.
    application: cold startproblem
    """
    # userID : int
    beerID : int
    reviewScore : float
    # reviewText : str
    reviewTime : datetime = Field(default_factory=datetime.now)
    # appearance : float
    # aroma : float
    # palate: float
    # taset: float
    # overall: float


class UserReviews(BaseModel):
    """
    description: 한 명의 유저가 평가한 맥주 리뷰 목록
    """
    userID: int
    review_beers: List[BeerReview] = Field(default_factory=list) 

    def add_reivew(self, single_beer_review: BeerReview):
        """
        description: 맥주 리뷰 목록에 맥주 리뷰를 추가
        """
        # TODO 이미 평가한 것을 재평가한 경우에는? 

        self.review_beers.append(single_beer_review)
        return self


class BeerRecommends(BaseModel):
    """
    description: 유저에게 보여줄 추천 맥주의 목록.
    """
    recommended_beers: List[Beer] = Field(default_factory=list) 

# 현재 사용되고 있지는 않는 부분
# class UserReviewsUpdate(BaseModel):
#     """
    
#     """
#     review_beers: List[BeerReview] = Field(default_factory=list) 