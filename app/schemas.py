from pydantic import BaseModel
from typing import Optional

class DetectionBox(BaseModel):
    category_code: str
    category_name: str
    confidence: float
    recyclable: bool
    severity_score: int
    sub_type_route: str
    weight_kg_estimate: float
    co2e_avoided_kg: float
    bbox_xyxy: list[float]
    count: Optional[int] = 1
    # INR value streams (v1.3)
    scrap_value_inr_min: float = 0.0
    scrap_value_inr_max: float = 0.0
    token_value_inr: float = 0.0
    carbon_credit_inr: float = 0.0
    total_value_inr_estimate: float = 0.0

class VolumeEstimate(BaseModel):
    method: str
    total_volume_m3: float
    density_kg_m3: dict[str, float]
    confidence: float

class ClassificationResult(BaseModel):
    request_id: str
    work_order_id: Optional[str]
    timestamp: str
    image_hash: str
    location_lat: Optional[float]
    location_lng: Optional[float]
    detections: list[DetectionBox]
    total_detections: int
    dominant_category: str
    dominant_category_name: str
    severity: str
    total_weight_kg_estimate: float
    total_co2e_avoided_kg: float
    categories_found: list[str]
    recyclable_fraction: float
    volume_estimate: VolumeEstimate
    ai_narrative: Optional[str] = None
    # Aggregate INR values (v1.3)
    total_scrap_value_inr_min: float = 0.0
    total_scrap_value_inr_max: float = 0.0
    total_token_value_inr: float = 0.0
    total_carbon_credit_inr: float = 0.0
    grand_total_value_inr: float = 0.0
