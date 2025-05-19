from pydantic import BaseModel


class StrictModel(BaseModel):
    class Config:
        extra = "forbid"
        frozen = True
