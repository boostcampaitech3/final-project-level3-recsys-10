from sqlalchemy import Column, ForeignKey, Integer, VARCHAR, DateTime, Float
from sqlalchemy.orm import relationship

from database import Base

from datetime import datetime

class User(Base):
    __tablename__ = "user"

    userID = Column(Integer, primary_key=True, index=True)
    password = Column(VARCHAR(20), nullable=False)
    profileName = Column(VARCHAR(50), nullable=False)
    gender = Column(VARCHAR(1), nullable=False)
    birth = Column(DateTime, nullable=False, default=datetime.now)

    review = relationship("Review", backref="user")

class Beer(Base):
    __tablename__ = "beer"

    beerId = Column(Integer, primary_key=True)
    beerName = Column(VARCHAR(100), nullable=False)
    brewerID = Column(Integer, nullable=False)
    ABV = Column(Float, nullable=True)
    style = Column(VARCHAR(50), nullable=False)
    imageUrl = Column(VARCHAR, nullable=True)

    review = relationship("Review", backref="beer")

class Review(Base):
    __tablename__ = "review"

    userID = Column(Integer, ForeignKey("user.userID"), primary_key=True)
    beerID = Column(Integer, ForeignKey("beer.beerId"), primary_key=True)
    reviewScore = Column(Float, nullable=False)
    reviewText = Column(VARCHAR, nullable=True)
    reviewTime = Column(DateTime, nullable=False, default=datetime.now)
    appearance = Column(Float, nullable=True)
    aroma = Column(Float, nullable=True)
    palate = Column(Float, nullable=True)
    taste = Column(Float, nullable=True)
    overall = Column(Float, nullable=True)