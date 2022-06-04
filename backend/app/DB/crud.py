# DB에 연결하여 직접적으로 Create(생성), Read(읽기), Update(갱신), Delete(삭제) 와 관련한 모듈들은 담당하는 곳
from typing import Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from sqlalchemy.sql import text

import numpy
import random
import pandas as pd

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

def get_beer_id(db: Session):
    return db.query(models.Beer.beer_id).all()

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

def get_beer_scores(db: Session, beer_id: int):
    s= f"""
    select avg(review_score), avg(appearance), avg(aroma), avg(palate), avg(taste), count(beer_id)
    from review
    where beer_id = {beer_id}
    """

    return list(db.execute(s).all()[0])

def get_popular_review(db: Session):
    s = """
    select beer_id, count(beer_id), avg(review_score)
    from review
    group by beer_id
    order by avg(review_score) desc
    """
    return db.execute(s).all()

def get_beer_review(db:Session, beer_id: int) -> List:
    s= f"""
    select u.profile_name, r.review_score, r.appearance, r.aroma, r.palate, r.taste, r.review_text
    from review as r
    join reviewer as u
    on r.user_id = u.user_id
    where r.beer_id = {beer_id};
    order by review_time desc
    """
    review = db.execute(s).all()
    return review

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

def get_coldstart_beer(db: Session):
    # beerID, style 을 불러옴
    beer_ids = []
    styles = []
    s = text(
    "SELECT beer_id,style FROM Beer"
    )
    for i in db.execute(s):
        beer_id, style = i
        beer_ids.append(beer_id)
        styles.append(style)

    id_dict = {id:style for id,style in zip(beer_ids,styles)}
    df = pd.DataFrame(list(id_dict.items()), columns=['beerId','style'])
    # coldstart ID 생성
    style_dict = {
        'Amber Lager - International / Vienna':0,
        'Apple Cider':1,
        'Belgian Ale - Dark / Amber':2,
        'Belgian Ale - Pale / Golden / Single':3,
        'Dark Lager - Dunkel / Tmavý':4,
        'Dark Lager - International / Premium':5,
        'Flavored - Fruit':6,
        'Flavored - Other':7,
        'IPA':8,
        'IPA - English':8,
        'Low / No Alcohol Beer - Pale':9,
        'Märzen / Oktoberfest Bier':10,
        'Pale Ale - American (APA)':11,
        'Pale Ale - English':11,
        'Pale Lager - American':12,
        'Pale Lager - International / Premium':12,
        'Pilsener - Bohemian / Czech':13,
        'Pilsener - Imperial':13,
        'Pilsener / Pils / Pilsner':13,
        'Radler / Shandy':14,
        'Stout':14,
        'Weissbier - Dunkelweizen':15,
        'Weissbier - Hefeweizen':15,
        'Wheat Ale':16,
        'Witbier / Belgian White Ale':17,
        'Zwickelbier / Kellerbier / Landbier':18
    }
    df['label'] = df['style'].apply(lambda x : style_dict[x])
    
    # return coldstart ID 
    ids = []
    for i in range(19):
        tmp = df[df['label']==i]
        if len(tmp) == 1:
            ids.extend(tmp['beerId'].values)
        elif len(tmp) > 1 and len(tmp) <10:
            ids.extend(numpy.random.choice(tmp['beerId'].values, 2,replace=False))
        else:
            ids.extend(numpy.random.choice(tmp['beerId'].values, 4,replace=False))
    
    ids = [int(i) for i in ids]
    items = db.query(models.Beer.beer_id, models.Beer.beer_name, models.Beer.image_url).filter(models.Beer.beer_id.in_(ids)).all()
    random.shuffle(items)  
    return items


def create_csscore(db: Session, 
                    submit: Dict,
                    user_id: int):
    max_id_before = db.query(func.max(models.ColdstartScore.csscore_id)).scalar()
    if max_id_before == None:
        max_id_before = 0

    for i, (beer_id_, cs_score) in enumerate(submit.items()):
        db_csscore = models.ColdstartScore(
                        csscore_id = int(max_id_before + 1 + i), 
                        user_id = user_id,
                        beer_id = beer_id_,
                        score = cs_score,
                    )
        db.add(db_csscore)
    db.commit()
    db.refresh(db_csscore)
    return db_csscore