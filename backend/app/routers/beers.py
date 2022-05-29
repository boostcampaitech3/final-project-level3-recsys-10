from fastapi import APIRouter

router = APIRouter()

@router.get("/beers/", tags=["beers"])
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]