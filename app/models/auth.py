from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, model_validator

from app.core.security import password_strength_errors


class UserPublic(BaseModel):
    id: uuid.UUID
    full_name: str
    email: EmailStr
    created_at: datetime


class GuestPublic(BaseModel):
    id: uuid.UUID
    expires_at: datetime
    created_at: datetime


class GuestCounts(BaseModel):
    portfolios: int = 0
    analyses: int = 0
    reports: int = 0


class AuthMeResponse(BaseModel):
    authenticated: bool
    user: UserPublic | None = None
    guest: GuestPublic | None = None
    guest_conversion_available: bool = False
    guest_counts: GuestCounts | None = None
    auth_session_expires_at: datetime | None = None
    guest_session_expires_at: datetime | None = None


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=200)
    email: EmailStr
    password: str = Field(..., min_length=10, max_length=256)
    confirm_password: str = Field(..., min_length=10, max_length=256)

    @model_validator(mode="after")
    def validate_passwords(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        errors = password_strength_errors(self.password)
        if errors:
            raise ValueError(" ".join(errors))
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=256)


class AuthActionResponse(BaseModel):
    user: UserPublic
    guest_conversion_available: bool = False
    guest_counts: GuestCounts | None = None


class GuestConvertRequest(BaseModel):
    confirm_merge: bool = False


class GuestConvertResponse(BaseModel):
    converted: bool
    already_converted: bool = False
    counts: GuestCounts = Field(default_factory=GuestCounts)


class MessageResponse(BaseModel):
    message: str
