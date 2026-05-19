from sqlalchemy import Column, Integer, String
from database import Base

class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String, unique=True)

    email = Column(String, unique=True)

    password = Column(String)

class DetectionHistory(Base):

    __tablename__ = "history"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    disease = Column(String)

    image = Column(String)

    created_at = Column(String)