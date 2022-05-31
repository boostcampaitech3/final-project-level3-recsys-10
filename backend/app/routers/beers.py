from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import main

from fastapi import Depends
from sqlalchemy.orm import Session
from ..DB.database import SessionLocal, engine
from ..DB import schemas, models, crud

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