# DB에 연결하여 직접적으로 Create(생성), Read(읽기), Update(갱신), Delete(삭제) 와 관련한 모듈들은 담당하는 곳

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from . import models
from . import schemas

def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.user_id == user_id).first()

def get_user_by_profile_name(db: Session, profile_name: str):
    return db.query(models.User).filter(models.User.profile_name == profile_name).first()

def get_user_id_by_profile_name(db: Session, profile_name: str):
    return db.query(models.User.user_id).filter(models.User.profile_name == profile_name).scalar()

def get_beer(db: Session, beer_id: int):
    return db.query(models.Beer).filter(models.Beer.beer_id == beer_id).first()

def update_feedback_by_id(db: Session, user_id: int, feedback_id: int, data_list: list):
    # db_feedback = db.query(models.Feedback).filter(models.Feedback.user_id == user_id).filter(models.Feedback.feedback_id == feedback_id)
    db_feedback = db.query(models.Feedback).filter(and_(models.Feedback.user_id == user_id, models.Feedback.feedback_id == feedback_id)).first()

    beer_id_list = [db_feedback.beer1_id, db_feedback.beer2_id, db_feedback.beer3_id, db_feedback.beer4_id]

    # 각각 맥주에 대한 피드백
    try:
        data_dict = {i.split("=")[0]:i.split("=")[1] for i in data_list}

        for idx, beer_id in enumerate(beer_id_list):
            try:
                score = data_dict[str(beer_id)]
                setattr(db_feedback, f"beer{idx+1}_score", int(score))
            except:
                continue
        
        # 조합에 대한 피드백
        try:
            score = data_dict['total']
            setattr(db_feedback, f"recommend", int(score))
        except:
            pass
        
        db.commit()
    except:
        pass

    return db_feedback

def get_popular_review(db: Session):
    s = """
    select beer_id, count(beer_id), avg(review_score)
    from review
    group by beer_id
    order by avg(review_score) desc
    """
    return db.execute(s).all()

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