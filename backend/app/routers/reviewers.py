from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse

from .. import main

from fastapi import Depends
from sqlalchemy.orm import Session
from ..DB.database import SessionLocal, engine
from ..DB import schemas, models, crud

from starlette.responses import RedirectResponse

router = APIRouter()

@router.get("/guide", response_class=HTMLResponse)
async def guide(request: Request):
    return main.templates.TemplateResponse("guide.html", {"request": request})

@router.post("/guide", response_class=HTMLResponse)
async def feedback(request: Request, user: list = Depends(main.get_current_user), db: Session = Depends(main.get_db)):
    feedback_data = await request.body()
    feedback_data = str(feedback_data, 'utf-8')
    feedback_data = feedback_data.split('&')

    print("-----feedback_data-----")
    print(feedback_data)

    user_id = crud.get_user_id_by_profile_name(db, profile_name=user[0])
    
    crud.update_feedback_by_id(db, user_id=user_id, feedback_id=user[1], data_list=feedback_data)
    
    return RedirectResponse(url="/guide", status_code=301)