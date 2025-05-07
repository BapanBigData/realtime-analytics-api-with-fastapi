from pydantic import BaseModel, Field
from typing import List


class RaceInput(BaseModel):
    race_year: int
    race_name: str
    driver_id: int
    driver_nationality: str
    team: str
    lap_percent: float
    position_gain: int
    pit_stop_rate: float
    rolling_avg_lap_time_3: float
    lap_time_cv_3: float
    qualifying_avg_sec: float
    qualifying_delta: float
    points: float
    rolling_fatigue_score: float
    position_change_rate_3: int
    is_pole_position: bool
    

class PreprocessingConfig(BaseModel):
    drop_columns: List[str] = Field(
        default=[
            'race_year',
            'race_name',
            'driver_id',
            'driver_nationality',
            'team',
            'points'
        ],
        description="Columns to drop as they are meta data and not useful for modeling"
    )

    features_to_scale: List[str] = Field(
        default=[
            'pit_stop_rate',
            'rolling_avg_lap_time_3',
            'lap_time_cv_3',
            'qualifying_avg_sec',
            'qualifying_delta',
            'rolling_fatigue_score'
        ],
        description="Numerical features that require scaling"
    )

    features_to_keep: List[str] = Field(
        default=[
            'lap_percent',
            'position_gain',
            'position_change_rate_3'
        ],
        description="Features to keep as is without scaling"
    )
    
    target_column: bool = Field(
        default='is_pole_position',
        description="Target column for prediction (binary classification)"
    )
