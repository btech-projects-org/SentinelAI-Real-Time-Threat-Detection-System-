
from pydantic import BaseModel, Field, field_validator
from fastapi import Form
from typing import Annotated

class CriminalCreate(BaseModel):
    criminal_id: str
    name: str = Field(min_length=2, max_length=100)
    threat_level: str = Field(default="MEDIUM")
    description: str = Field(default="")

    @field_validator('criminal_id')
    @classmethod
    def validate_id(cls, v: str):
        if not v.isalnum() and "_" not in v and "-" not in v:
             raise ValueError('ID must be alphanumeric (dash/underscore allowed)')
        return v

    @classmethod
    def as_form(
        cls,
        criminal_id: str = Form(...),
        name: str = Form(...),
        threat_level: str = Form(default="MEDIUM"),
        description: str = Form(default="")
    ):
        return cls(
            criminal_id=criminal_id,
            name=name,
            threat_level=threat_level,
            description=description
        )
