from __future__ import annotations
from typing import Literal, List
from pydantic import BaseModel, Field

# ---------- Core request/response models ----------


class UserProfile(BaseModel):
    user_id: str
    age: int = Field(ge=13, le=100)
    gender: Literal["male", "female", "other"]
    work_pattern: Literal["9-5", "shift", "flex"]
    primary_goal: Literal["weight_loss", "stress", "fitness"]
    baseline_activity_min_per_day: int = Field(ge=0, le=300)
    premium: bool
    push_opt_in: bool
    chronotype: Literal["morning", "evening"]
    language: Literal["en", "de", "fr"]

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "u0001",
                "age": 29,  # non-zero placeholder
                "gender": "female",
                "work_pattern": "9-5",
                "primary_goal": "weight_loss",
                "baseline_activity_min_per_day": 12,  # non-zero placeholder
                "premium": True,
                "push_opt_in": True,
                "chronotype": "morning",
                "language": "en",
            }
        }


class RequestContext(BaseModel):
    day_of_week: int = Field(ge=0, le=6)  # 0=Mon ... 6=Sun
    hour_bucket: Literal["morning", "evening"]

    class Config:
        json_schema_extra = {"example": {"day_of_week": 6, "hour_bucket": "morning"}}


class RecommendationRequest(BaseModel):
    user: UserProfile
    context: RequestContext
    top_k: int = Field(default=5, ge=1, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "user": UserProfile.Config.json_schema_extra["example"],
                "context": RequestContext.Config.json_schema_extra["example"],
                "top_k": 5,
            }
        }


class RecommendationItem(BaseModel):
    content_id: str
    type: str
    duration_min: int
    intensity: str
    goal_tag: str
    difficulty: str
    score: float


class RecommendationResponse(BaseModel):
    persona: int
    chosen_arm: str
    items: List[RecommendationItem]
    rationale: str


class Feedback(BaseModel):
    user_id: str
    content_id: str
    arm: Literal[
        "push_morning",
        "push_evening",
        "email_morning",
        "email_evening",
        "inapp_morning",
        "inapp_evening",
    ]
    reward: int = Field(ge=0, le=1)
    day_of_week: int = Field(ge=0, le=6)
    hour_bucket: Literal["morning", "evening"]

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "u0001",
                "content_id": "c0002",
                "arm": "push_morning",
                "reward": 1,  # non-zero placeholder
                "day_of_week": 6,
                "hour_bucket": "morning",
            }
        }


# ---------- Single consolidated helper model ----------


class HelperBundle(BaseModel):
    arms: List[str]
    sample_user: UserProfile
    sample_content_id: str
    recommend_example: RecommendationRequest
    feedback_example: Feedback
