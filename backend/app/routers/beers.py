import random

from jose import jwt
from fastapi import Depends, APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from .. import main
from ..DB.database import SessionLocal
from ..DB import models, crud
from ...recommendAPI.model import AutoRec, get_model , predict_from_select_beer, popular_topk
from ...recommendAPI.s3rec.inference_api import inference
from backend.recommendAPI.s3rec.inference_api import inference


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
    s = """
    SELECT targetbeer.beer_id, beer.beer_name
    FROM targetbeer
    INNER JOIN beer
    ON targetbeer.beer_id=beer.beer_id
    LIMIT 10
    """
    beers = [(
        beer_id, 
        beer_name,
        f"https://raw.githubusercontent.com/minchoul2/beer_image/main/beer_image/{beer_id}.png") 
            for beer_id, beer_name in db.execute(s).all()]
    return main.templates.TemplateResponse("index.html", {"request": request, "beers": beers})

@router.get("/coldstart", response_class=HTMLResponse)
async def coldstart(request: Request, db: Session = Depends(get_db)):
    beers= crud.get_coldstart_beer(db)
    return main.templates.TemplateResponse("coldstart.html", {"request": request, "beers": beers})

@router.get("/beer", response_class=HTMLResponse)
async def beerList(request: Request, db: Session = Depends(get_db)):
    s = """
    SELECT targetbeer.beer_id, beer.beer_name
    FROM targetbeer
    INNER JOIN beer
    ON targetbeer.beer_id=beer.beer_id
    """
    beers = [(
        beer_id, 
        beer_name,
        f"https://raw.githubusercontent.com/minchoul2/beer_image/main/beer_image/{beer_id}.png") 
            for beer_id, beer_name in db.execute(s).all()]

    return main.templates.TemplateResponse("beerList.html", {"request": request, "beers": beers})

@router.post("/result")
async def prefer(request: Request, 
                user: list = Depends(main.get_current_user), 
                db: Session = Depends(get_db),
                model : AutoRec = Depends(get_model)):
    data_b = await request.body()
    data_b = str(data_b, 'utf-8')
    data_b = data_b.split('&') # 리스트
    user_id = crud.get_user_id_by_profile_name(db, profile_name=user[0])

    recommend_choice = random.randrange(1,101)
    try:
        data_dict = {}
        for i in data_b:
            beer_id, rate = i.split('=')
            data_dict[int(beer_id)] = int(rate)
        crud.create_csscore(db, submit = data_dict, user_id = user_id)

        if recommend_choice % 2 == 0:
            # AutoRec
            # topk_pred = predict_from_select_beer(model, data_dict)
            
            # S3Rec
            print('~~~~~~~~모델 기반 추천~~~~~~~~~~~~~~~')
            beer_ids = crud.get_target_beer_id(db)  # Filtering target beer id 
            topk_pred = inference(data_dict, beer_ids)
            recommend_type = 0
        else :
            print('~~~~~~~~인기도 기반 추천~~~~~~~~~~~~~~~')
            data = crud.get_popular_review(db)
            topk_pred = popular_topk(data, topk=4, method='steam')
            recommend_type = 1
    except:
        # 인기도 기반 추천
        print('~~~~~~~~예외 ) 인기도 기반 추천 ~~~~~~~~~~~~~~~')
        recommend_type = 1
        data = crud.get_popular_review(db)
        topk_pred = popular_topk(data, topk=4, method='steam')

    RecommendedBeer_1 = crud.get_beer_for_recommend(db, beer_id = int(topk_pred[0])) # Beer
    RecommendedBeer_2 = crud.get_beer_for_recommend(db, beer_id = int(topk_pred[1]))
    RecommendedBeer_3 = crud.get_beer_for_recommend(db, beer_id = int(topk_pred[2]))
    RecommendedBeer_4 = crud.get_beer_for_recommend(db, beer_id = int(topk_pred[3]))
        
    RecommendedBeer_1.abv = round(RecommendedBeer_1.abv, 1) # Beer
    RecommendedBeer_2.abv = round(RecommendedBeer_2.abv, 1)
    RecommendedBeer_3.abv = round(RecommendedBeer_3.abv, 1)
    RecommendedBeer_4.abv = round(RecommendedBeer_4.abv, 1)


    # result 결과 => DB 저장
    max_id_before = db.query(func.max(models.Feedback.feedback_id)).filter(models.Feedback.user_id == user_id).scalar()
    
    if max_id_before == None:
        max_id_before = 0
    db_feedback = models.Feedback(
                feedback_id=int(max_id_before + 1),
                user_id=user_id, 
                recommend_type = recommend_type,
                recommend=None,
                beer1_id=int(topk_pred[0]),
                beer2_id=int(topk_pred[1]),
                beer3_id=int(topk_pred[2]),
                beer4_id=int(topk_pred[3]), 
                beer1_score=None,
                beer2_score=None,
                beer3_score=None,
                beer4_score=None
            )

    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    token = jwt.encode({"nickname": user[0], "feedback_id": int(max_id_before + 1)}, main.secret_key)
    response = main.templates.TemplateResponse("result.html", {"request": request, "beers": [RecommendedBeer_1, RecommendedBeer_2, RecommendedBeer_3, RecommendedBeer_4]})
    response.set_cookie("session", token)

    return response
