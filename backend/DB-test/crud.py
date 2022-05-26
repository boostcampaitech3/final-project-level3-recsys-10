from sqlalchemy.orm import Session

import models
import schemas

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.userid == user_id).first()

def get_beer(db: Session, beer_id: int):
    return db.query(models.Beer).filter(models.Beer.beerid == beer_id).first()

def create_user(db: Session, user: schemas.UserCreate):
    # fake_hashed_password = user.password + "notreallyhashed"
    fake_hashed_password = user.password
    db_user = models.User(user_id=user.userID, password=fake_hashed_password, profile_name=user.profileName,
                        gender=user.gender, birth=user.birth)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user