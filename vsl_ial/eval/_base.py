from pydantic import BaseModel


class StrictModel(BaseModel, extra="forbid"):
    pass
