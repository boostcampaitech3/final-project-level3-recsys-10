from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse

from .. import main

from fastapi import Depends
from sqlalchemy.orm import Session
from ..DB.database import SessionLocal, engine
from ..DB import schemas, models, crud

from starlette.responses import RedirectResponse

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/beer/{beer_id}", response_class=HTMLResponse)
async def beer(request: Request, beer_id: int, db: Session = Depends(get_db)):
    beer = crud.get_beer(db, beer_id=beer_id)
    beerInfo = [beer.beer_name, beer.abv, beer.image_url]
    # TODO
    '''
    description : reviews 객체에 있는 user_id를 통해 reviewer 테이블의 profile_name 받아오기(by. join)
    input : reviews 객체
    output : reviewer.profile_name

    -> 원래는 DB 객체를 받아오는 형태(profilename미포함)에서
       DB에서 필요한 데이터(profilename포함)를 리스트로 받아오는 형태로 작성.
       [profile_name, reviewscore, appearance, aroma, palate, taste, reviewtext] 순
    '''
    # reviews = db.query(models.Review).filter(models.Review.beer_id == beer_id).all()
    reviews = crud.get_beer_review(db, beer_id)
  

    return main.templates.TemplateResponse("beer.html", {"request": request, "beerInfo": beerInfo, "reviews": reviews})

@router.post("/beer/{beer_id}", response_class=HTMLResponse)
async def beerEvaluation(beer_id: int, appearance: int = Form(...), aroma: int = Form(...),
                        palate: int = Form(...), taste: int = Form(...), comment: list = Form(...), 
                        nickname: str = Depends(main.get_current_user), db: Session = Depends(get_db)):

    review_score = (appearance + aroma + palate + taste) // 4

    user = crud.get_user_by_profile_name(db, profile_name=nickname)

    db_review = models.Review(
                 user_id=user.user_id, 
                 beer_id=beer_id,
                 review_score=review_score,
                 review_text=comment[0],
                 appearance=appearance,
                 aroma=aroma, 
                 palate=palate,
                 taste=taste,
                 overall=0
            )

    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return RedirectResponse(url="/beer/"+str(beer_id), status_code=301)