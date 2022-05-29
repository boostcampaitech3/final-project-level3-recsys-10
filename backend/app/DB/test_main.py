from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

import crud
import models
import schemas
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 회원가입
@app.post("/register", response_model=schemas.User)
def signup_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_id(db, user_id=user.user_id)
    if db_user:
        raise HTTPException(status_code=400, detail="Error 발생!")
    return crud.create_user(db=db, user=user)

# 로그인
@app.get("/login/{user_id}", response_model=schemas.User)
def login_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_id(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

# 맥주 정보 출력
@app.get("/beer/{beer_id}", response_model=schemas.Beer)
def read_beer(beer_id: int, db: Session = Depends(get_db)):
    db_beer = crud.get_beer(db, beer_id=beer_id)
    if db_beer is None:
        raise HTTPException(status_code=404, detail="Beer not found")
    return db_beer