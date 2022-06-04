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
    '''
    description : reviews 객체에 있는 user_id를 통해 reviewer 테이블의 profile_name 받아오기(by. join)
    input : reviews 객체
    output : reviewer.profile_name

    -> 원래는 DB 객체를 받아오는 형태(profilename미포함)에서
       DB에서 필요한 데이터(profilename포함)를 리스트로 받아오는 형태로 작성.
       [profile_name, reviewscore, appearance, aroma, palate, taste, reviewtext] 순
    '''
    reviews = crud.get_beer_review(db, beer_id)
    avg_review_score, avg_appearence, avg_aroma, avg_palate, avg_tasta, cnt_reviews = crud.get_beer_scores(db, beer_id=beer_id)
    
    return main.templates.TemplateResponse("beer.html", 
                                            {"request": request, "beerInfo": beerInfo, "reviews": reviews,
                                             "avg_review_score":avg_review_score, "avg_appearence":avg_appearence,
                                             "avg_aroma":avg_aroma, "avg_palate":avg_palate,"avg_tasta":avg_tasta,
                                             "cnt_reviews":cnt_reviews})

@router.post("/beer/{beer_id}", response_class=HTMLResponse)
async def beerEvaluation(beer_id: int, appearance: int = Form(...), aroma: int = Form(...),
                        palate: int = Form(...), taste: int = Form(...), comment: list = Form(...), 
                        user: list = Depends(main.get_current_user), db: Session = Depends(get_db)):

    review_score = (appearance + aroma + palate + taste) // 4

    user = crud.get_user_by_profile_name(db, profile_name=user[0])

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

@router.get("/guide", response_class=HTMLResponse)
async def guide(request: Request):
    return main.templates.TemplateResponse("guide.html", {"request": request})

@router.post("/guide", response_class=HTMLResponse)
async def feedback(request: Request, user: list = Depends(main.get_current_user), db: Session = Depends(main.get_db)):
    feedback_data = await request.body()
    feedback_data = str(feedback_data, 'utf-8')
    feedback_data = feedback_data.split('&')

    user_id = crud.get_user_id_by_profile_name(db, profile_name=user[0])
    
    crud.update_feedback_by_id(db, user_id=user_id, feedback_id=user[1], data_list=feedback_data)
    
    return RedirectResponse(url="/guide", status_code=301)
