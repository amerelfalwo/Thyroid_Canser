from pydantic import BaseModel, Field

class MetadataInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: int = Field(..., ge=0, le=1)
    TIRADS: int = Field(..., ge=1, le=5)
    FNAC: int = Field(..., ge=0, le=6)
