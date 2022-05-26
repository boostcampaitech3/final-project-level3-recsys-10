from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import yaml

with open('db_config.yaml') as f:
    setting = yaml.safe_load(f)
    user = setting['user']
    password = setting['password']

SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://" + user + ":" + password + "@0.0.0.0:5432/beerrecsysdb"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()