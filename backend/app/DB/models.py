from email.policy import default
from sqlalchemy import Column, ForeignKey, Integer, VARCHAR, DateTime, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.app.DB.database import Base

 
class User(Base): 
    __tablename__ = "reviewer"

    user_id = Column(Integer, primary_key=True)
    password = Column(VARCHAR(35), nullable=False)
    profile_name = Column(VARCHAR(50), nullable=False)
    gender = Column(VARCHAR(1), nullable=True)
    birth = Column(DateTime, nullable=True, default=datetime.now)

    review = relationship("Review", backref="reviewer")

class Beer(Base):
    __tablename__ = "beer"

    beer_id = Column(Integer, primary_key=True)
    beer_name = Column(VARCHAR(100), nullable=False)
    brewer_id = Column(Integer, nullable=False)
    abv = Column(Float, nullable=True)
    style = Column(VARCHAR(50), nullable=False)
    image_url = Column(VARCHAR, nullable=True)
    review = relationship("Review", backref="beer")

class Review(Base):
    __tablename__ = "review"

    user_id = Column(Integer, ForeignKey("reviewer.user_id"), primary_key=True)
    beer_id = Column(Integer, ForeignKey("beer.beer_id"), primary_key=True)
    review_score = Column(Float, nullable=False)
    review_text = Column(VARCHAR, nullable=True)
    review_time = Column(DateTime, nullable=False, default=datetime.now)
    appearance = Column(Float, nullable=True)
    aroma = Column(Float, nullable=True)
    palate = Column(Float, nullable=True)
    taste = Column(Float, nullable=True)
    overall = Column(Float, nullable=True)

class Feedback(Base):
    __tablename__ = "feedback"

    feedback_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("reviewer.user_id"), primary_key=True)
    recommend_type = Column(Integer, nullable = True)
    recommend = Column(Integer, nullable = True)
    beer1_id = Column(Integer, nullable = False)
    beer2_id = Column(Integer, nullable = False)
    beer3_id = Column(Integer, nullable = False)
    beer4_id = Column(Integer, nullable = False)
    beer1_score = Column(Integer, nullable = True)
    beer2_score = Column(Integer, nullable = True)
    beer3_score = Column(Integer, nullable = True)
    beer4_score = Column(Integer, nullable = True)

class ColdstartScore(Base):
    __tablename__ = "csscore"

    csscore_id = Column(Integer, primary_key = True)
    user_id = Column(Integer, ForeignKey("reviewer.user_id"))
    beer_id = Column(Integer, ForeignKey("beer.beer_id"))
    score = Column(Integer, nullable = True)

