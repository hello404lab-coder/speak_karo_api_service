"""Request and response schemas for onboarding endpoints."""
from typing import Literal

from pydantic import BaseModel, Field


class OnboardingStatusResponse(BaseModel):
    """Response for GET /api/v1/auth/onboarding/status."""

    onboarding_completed: bool = Field(..., description="Whether user finished onboarding")
    current_step: int = Field(..., description="Current onboarding step (0-5)")


class NicknameStep(BaseModel):
    """Step 1: nickname (3-30 characters)."""

    nickname: str = Field(..., min_length=3, max_length=30, description="Display nickname")


class NativeLanguageStep(BaseModel):
    """Step 2: native language."""

    native_language: str = Field(..., min_length=1, max_length=100, description="Native language")


class StudentDetailsStep(BaseModel):
    """Step 3: student type and occupation."""

    student_type: Literal["adult", "kid"] = Field(..., description="Student type")
    occupation: Literal["college", "work", "home_maker", "teacher", "other"] = Field(
        ..., description="Occupation"
    )


class GoalStep(BaseModel):
    """Step 4: reason for using the app."""

    goal: Literal[
        "prepare_interviews",
        "prepare_govt_exams",
        "go_abroad",
        "talk_family",
        "build_confidence",
        "improve_workplace",
        "other",
    ] = Field(..., description="What brings the user to the app")


class EnglishLevelStep(BaseModel):
    """Step 5: current English level."""

    english_level: Literal[
        "beginner",
        "elementary",
        "intermediate",
        "upper_intermediate",
        "advanced",
    ] = Field(..., description="Current English level")
