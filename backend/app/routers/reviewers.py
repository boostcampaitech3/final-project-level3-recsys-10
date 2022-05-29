from fastapi import APIRouter

router = APIRouter()

@router.get("/reviewers/", tags=["reviewers"])
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]