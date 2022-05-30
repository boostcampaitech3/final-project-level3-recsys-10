# DB에 연결하여 직접적으로 Create(생성), Read(읽기), Update(갱신), Delete(삭제) 와 관련한 모듈들은 담당하는 곳
import pandas as pd
import numpy 
import random

from sqlalchemy.orm import Session
from sqlalchemy import func

from . import models
from . import schemas

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
    return db_user\

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
    random.shuffle(ids)  

    return ids