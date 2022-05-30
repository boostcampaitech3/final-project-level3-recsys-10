from fastapi import APIRouter, Request, Form, Response
from fastapi.responses import HTMLResponse

from .. import main

from fastapi import Depends
from sqlalchemy.orm import Session
from ..DB.database import SessionLocal, engine
from ..DB import schemas, models

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    beers = db.query(models.Beer.beer_name, models.Beer.abv, models.Beer.image_url).limit(6).all()
    return main.templates.TemplateResponse("index.html", {"request": request, "beers": beers})

@router.get("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, db: Session = Depends(get_db)):
    beers = db.query(models.Beer.beer_id, models.Beer.beer_name, models.Beer.image_url).limit(30).all()
    return main.templates.TemplateResponse("recommend.html", {"request": request, "beers": beers})

@router.post(path="/recommend")
async def prefer(request: Request, response: Response, call_next, db: Session = Depends(get_db)):
    # DB에 데이터 저장 및 모델에 사용자 정보 넘기기
    # response = await call_next(request)
    print("----------start------------")
    data_b = await request.body()
    print("-----------------data_b--------------------")
    print(data_b)
    result = vehicle_detector.detect(data_b)
    print("--------result--------")
    print(result)
    return JSONResponse(result)
    # print(formData)
    # return RedirectResponse(url="/recommendResult", status_code=301)