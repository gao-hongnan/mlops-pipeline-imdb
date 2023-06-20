# pydantic does not gel well with linting.
# pylint: disable=no-self-argument,no-name-in-module
from typing import List
from pydantic import BaseModel, Field, validator


class RatingInput(BaseModel):
    """Corresponds to the input data for the API."""

    genres: List[str] = Field(..., description="The genres of the movie")
    titles: List[str] = Field(..., description="The titles of the movie")


class Rating(BaseModel):
    """Internal representation of a rating."""

    text: str = Field(..., description="The review text")
    rating: int = Field(..., description="The rating of the movie")
    probability: List[float] = Field(..., description="The probability of the rating")

    @validator("probability", each_item=True)
    def probability_must_be_in_range(cls, value):
        if not 0 <= value <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        return value


class RatingOutput(BaseModel):
    ratings: List[Rating]

    class Config:
        schema_extra = {
            "example": {
                "ratings": [
                    {
                        "text": "The Maze-Action_Sci-Fi",
                        "rating": 0,
                        "probability": [0.50204438, 0.49795562],
                    },
                    {
                        "text": "The Godfather is here in Singapore-Crime_Drama",
                        "rating": 1,
                        "probability": [0.32124885, 0.67875115],
                    },
                    {
                        "text": "Maybe he loves you-Love_Crime",
                        "rating": 1,
                        "probability": [0.23760057, 0.76239943],
                    },
                ]
            }
        }
