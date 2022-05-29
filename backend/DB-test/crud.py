from sqlalchemy.orm import Session
from sqlalchemy import func

import models
import schemas

def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.user_id == user_id).first()

def get_user_by_profile_name(db: Session, profile_name: str):
    return db.query(models.User).filter(models.User.profile_name == profile_name).first()

def get_beer(db: Session, beer_id: int):
    return db.query(models.Beer).filter(models.Beer.beer_id == beer_id).first()

def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"    
    max_id_before = db.query(func.max(models.User.user_id)).scalar()
    db_user = models.User(
                    user_id=int(max_id_before + 1), 
                    password=fake_hashed_password, 
                    profile_name=user.profile_name,
                    gender=user.gender, 
                    birth=user.birth
                )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user