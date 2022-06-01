from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import main

from fastapi import Depends
from sqlalchemy.orm import Session
from ..DB.database import SessionLocal, engine
from ..DB import schemas, models,crud
from ...recommendAPI.model import AutoRec, get_model , predict_from_select_beer, popular_topk

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/index", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    beers = db.query(models.Beer.beer_name, models.Beer.abv, models.Beer.image_url).limit(6).all()
    return main.templates.TemplateResponse("index.html", {"request": request, "beers": beers})

@router.get("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, db: Session = Depends(get_db)):
    beers = db.query(models.Beer.beer_id, models.Beer.beer_name, models.Beer.image_url).limit(30).all()
    return main.templates.TemplateResponse("recommend.html", {"request": request, "beers": beers})

@router.post("/result")
async def prefer(request: Request, 
                db: Session = Depends(get_db),
                model : AutoRec = Depends(get_model)):
    data_b = await request.body()
    data_b = str(data_b, 'utf-8')
    data_b = data_b.split('&') # 리스트

    try:
        data_dict = {}
        for i in data_b:
            beer_id, rate = i.split('=')
            data_dict[int(beer_id)] = int(rate)
            topk_pred, topk_rating = predict_from_select_beer(model, data_dict)
    except:
        # TODO : 인기도 기반 추천
        data = crud.get_popular_review(db)
        topk_pred = popular_topk(data, topk=4, method='count')
        print("----------------------pop-----------------------------")
        print(topk_pred)
    
    RecommendedBeer_1 = crud.get_beer(db, beer_id = int(topk_pred[0])) # Beer
    RecommendedBeer_2 = crud.get_beer(db, beer_id = int(topk_pred[1]))
    RecommendedBeer_3 = crud.get_beer(db, beer_id = int(topk_pred[2]))
    RecommendedBeer_4 = crud.get_beer(db, beer_id = int(topk_pred[3]))
    
    return main.templates.TemplateResponse("result.html", {"request": request, "beers": [RecommendedBeer_1, RecommendedBeer_2, RecommendedBeer_3, RecommendedBeer_4]})