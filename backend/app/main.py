from fastapi import FastAPI, Request
from fastapi.param_functions import Depends
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from datetime import datetime
from ..recommendAPI.model import AutoRec, get_model , predict_from_select_beer
from .routers import users, beers, reviewers

from sqlalchemy.orm import Session
import backend.app.DB.crud as crud
import backend.app.DB.schemas as schemas
from backend.app.DB.database import SessionLocal, engine
import backend.app.DB.models as models

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 

# DB 서버에 연결
models.Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

app.include_router(users.router)
app.include_router(beers.router)
app.include_router(reviewers.router)

templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

class Product(BaseModel):
    id: str
    score: float

class Rec_Product(Product):
    img : str

class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    #products: List[Product] = Field(default_factory=list)
    products: List[Rec_Product] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_product(self, product: Product):
        if product.id in [exisiting_product.id for exisiting_product in self.products]:
            return self
        
        self.products.append(product)
        self.updated_at = datetime.now()
        return self

class InferenceRecProduct(Product):
    name: str = "inference_rec_product"
    score: float = 100.0
    result: Optional[List]


# - [ ]  맥주 리스트를 유저에게 보여주기(3~10개?)
#     - [ ]  해당 맥주의 이미지 및 이름 보여주기
# - [ ]  유저가 선호하는 맥주 리스트 받기
#     - [ ]  리스트를 받음과 동시에 inference 요청
#     - [ ]  PostgreSQL 맥주 리스트 업로드
# - [ ]  inference의 결과값을 유저에게 보여주기 : 4캔 조합
#     - [ ]  4개의 맥주 이미지 및 이름 보여주기

# @app.get("/order", description="유저가 맥주를 선택합니다.")
# def get_orders() -> List[Order]:
#     return product_list

# @app.get("/order/{order_id}", description="유저가 맥주를 선택합니다.")
# def get_order(order_id: int) -> List[Order]:
#     new_product = Product(name=product_list[order_id], score=order_id)
#     #new_product.name = get_order_by_id(order_id)
#     #new_order = Order(products=[new_product])
#     return new_product

# def get_order_by_id(order_id: int):
#     return product_list[order_id]

# @app.post("/select", description="유저가 선호하는 맥주를 선택합니다")
# def preference_select(products : dict):
#     beer_list = []
#     for key,value in enumerate(products):
#         user_beer =  Rec_Product(id=value, score=key, img="imgs")
#         beer_list.append(user_beer)
#     order = Order(products=beer_list)
#     return order

# @app.post("/select", description="유저가 선호하는 맥주를 선택합니다",  response_model=dict)
# def preference_select(products : dict,
#                       model : AutoRec = Depends(get_model)):

#     topk_pred, topk_rating = predict_from_select_beer(model, products)
#     dic = {str(name):value for name, value in zip(topk_pred,topk_rating)}
#     return dic

@app.post("/select", description="유저가 선호하는 맥주를 선택합니다", response_model=List[schemas.Beer])
def preference_select(products : dict,
                      model : AutoRec = Depends(get_model),
                      db: Session = Depends(get_db)):

    topk_pred, topk_rating = predict_from_select_beer(model, products)
    
    RecommendedBeer_1 = crud.get_beer(db, beer_id = int(topk_pred[0])) # Beer
    RecommendedBeer_2 = crud.get_beer(db, beer_id = int(topk_pred[1]))
    RecommendedBeer_3 = crud.get_beer(db, beer_id = int(topk_pred[2]))
    RecommendedBeer_4 = crud.get_beer(db, beer_id = int(topk_pred[3]))

    # print(">>>>", RecommendedBeer_1.beer_id)

    return [RecommendedBeer_1, RecommendedBeer_2, RecommendedBeer_3, RecommendedBeer_4]