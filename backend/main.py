from itertools import product
from fastapi import FastAPI
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from datetime import datetime

app = FastAPI()

product_list = ["Sapporo Premium Beer / Draft Beer ", 
"Tsingtao Premium Stout 4.8%",
"Tsingtao Draft Beer 11º (Pure Draft Beer)",
"Heineken",
"Heineken Dark Lager",
"Heineken Premium Light"]

product_images_list = ["https://www.ratebeer.com/beer/sapporo-premium-beer-draft-beer/729/](https://www.ratebeer.com/beer/sapporo-premium-beer-draft-beer/729/",

"https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_567803](https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_567803",

"https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_64518](https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_64518",

"https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_37](https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_37",

"https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_34662](https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_34662",

"https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_48076](https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_48076"]

rec_list = ["Cass Fresh",
"Cass light"
]

@app.get("/")
def hello_world():
    return {"hello": "world"}

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

@app.post("/select", description="유저가 선호하는 맥주를 선택합니다")
def preference_select(products : dict):
    beer_list = []
    for key,value in enumerate(products):
        user_beer =  Rec_Product(id=value, score=key, img="imgs")
        beer_list.append(user_beer)
    order = Order(products=beer_list)
    return order

# @app.post("/order/", description="맥주 추천을 요청합니다.")
# def make_order() -> Order:
    
#     products =[]
#     for _ in product_list:
#         Inference_result = rec_list
#         product = InferenceRecProduct(result=Inference_result)
#         products.append(product)

#     new_order = Order(products=products)

    
#     return new_order