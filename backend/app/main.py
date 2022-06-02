from fastapi import FastAPI, Request, Form, Cookie, Security 

from fastapi.param_functions import Depends
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from typing import List, Optional

from datetime import datetime
from ..recommendAPI.model import AutoRec, get_model , predict_from_select_beer

from sqlalchemy.orm import Session
from sqlalchemy.sql import text
import backend.app.DB.crud as crud
import backend.app.DB.schemas as schemas
from backend.app.DB.database import SessionLocal, engine
import backend.app.DB.models as models

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 

from starlette.responses import Response, HTMLResponse, RedirectResponse

from fastapi.security import APIKeyCookie
from jose import jwt

import yaml

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

cookie_sec = APIKeyCookie(name="session")

with open('backend/app/config.yaml') as f:
    setting = yaml.safe_load(f)
    secret_key = setting['secret_key']
secret_key = secret_key

def get_current_user(session: str = Depends(cookie_sec)):
    try:
        payload = jwt.decode(session, secret_key)
        user = payload["sub"]
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid authentication"
        )

from .routers import users, beers, reviewers

app.include_router(users.router)
app.include_router(beers.router)
app.include_router(reviewers.router)

templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("nickname_login.html", {"request": request})

@app.post("/", description="Insert profile_name", response_class=HTMLResponse)
async def login_check(request: Request, response: Response, nickname: str = Form(...), db: Session = Depends(get_db)):
    # 이미 존재하는 닉네임인지 확인하는 작업
    isexist = crud.get_user_by_profile_name(db, profile_name = nickname)
    if isexist:
        return templates.TemplateResponse("nickname_login.html", {"request": request, "error": True})

    # 새로운 유저 정보 등록
    new_user = schemas.UserCreate()
    new_user.profile_name = nickname # user_id, gender, birth 아직은 관련 정보를 받지 않을 예정, 그러나 birth는 아이디 생성 시간으로 기록될 예정
    new_user.gender = "X"
    new_user.password = "BoostcampOnlineTest"
    crud.create_user(db, user = new_user)

    token = jwt.encode({"sub": nickname}, secret_key)
    response = RedirectResponse(url="/index", status_code=301)
    response.set_cookie("session", token)

    return response


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

    return [RecommendedBeer_1, RecommendedBeer_2, RecommendedBeer_3, RecommendedBeer_4]

@app.post("/coldstart", description= "유저에게 보여줄 맥주의 리스트를 보여줍니다", response_model=List[schemas.Beer])
def showing_coldstart(db: Session = Depends(get_db)):
    coldstart_beers  = crud.get_coldstart_beer(db)
    return coldstart_beers