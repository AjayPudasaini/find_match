from fastapi import FastAPI
from database.db import engine
from database.db import Base
from router.find_match import router as find_match_router
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(find_match_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

