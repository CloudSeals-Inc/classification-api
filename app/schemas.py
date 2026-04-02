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
