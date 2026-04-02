"""
MIBA — Classification API v1.2
Added: /classify/debug endpoint, explicit error logging in enrichment loop
"""

import os, io, base64, hashlib, logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from .taxonomy import INDICWASTE, get_by_code, TOKEN_INR_RATES
from .model import ClassificationModel
from .schemas import ClassificationResult, DetectionBox, VolumeEstimate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MIBA Classification API", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*", "https://miba-ui-460918627115.europe-west1.run.app"],
                  allow_methods=["*"], allow_headers=["*"])

_model: Optional[ClassificationModel] = None

def get_model() -> ClassificationModel:
    global _model
    if _model is None:
        _model = ClassificationModel(
            yolo_weights=os.getenv("YOLO_WEIGHTS_PATH", "weights/yolov9_indicwaste.pt"),
            efficientnet_weights=os.getenv("EFFICIENTNET_WEIGHTS_PATH",
                                           "weights/efficientnet_b3_indicwaste.pt"),
            midas_weights=os.getenv("MIDAS_WEIGHTS_PATH", "weights/midas_small.pt"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.45")),
            device=os.getenv("INFERENCE_DEVICE", "cpu"),
            vertex_endpoint_id=os.getenv("VERTEX_ENDPOINT_ID"),
        )

    return _model

# Severity for dominant scoring (W16 penalised)
_SEV = {"W01":5,"W02":8,"W03":9,"W04":6,"W05":7,"W06":4,"W07":3,
        "W08":10,"W09":10,"W10":6,"W11":7,"W12":3,"W13":10,"W14":8,"W15":2,"W16":5}


@app.get("/health")
def health(model: ClassificationModel = Depends(get_model)):
    return {"status": "ok", "version": "1.2.0",
            "stub_mode": model._stub_mode, "device": model.device}


def _enrich(detections: list[dict], classifications: list[dict],
            volume_est: VolumeEstimate) -> list[DetectionBox]:
    """
    Merge detection + classification + taxonomy into DetectionBox list.
    Logs every drop so failures are never silent.
    """
    boxes = []
    for i, (det, cls_r) in enumerate(zip(detections, classifications)):
        code = cls_r.get("category_code", "")
        cat = get_by_code(code)
        
        # If taxonomy lookup fails, use a generic default instead of dropping
        if cat is None:
            from .schemas import WasteCategory
            cat = WasteCategory(
                code="W16",
                name="Mixed Waste",
                recyclable=False,
                severity_score=5,
                co2e_avoided_per_tonne=0
            )
            logger.info("ENRICH: Code %r not in taxonomy, using default W16", code)
        density = volume_est.density_kg_m3.get(code, 150)
        vol_m3  = det.get("volume_m3", 0.001)
        weight  = density * vol_m3
        co2e    = round(weight * cat.co2e_avoided_per_tonne / 1000, 4)
        name = det.get("category_name") or cat.name

        # INR value streams (v1.3)
        # scrap: conservative min-rate estimate
        # token: weight × TOKEN_INR_RATE × 0.5 (citizen-facing token allocation)
        # carbon: ₹1/kg fixed simple estimate
        scrap_min  = round(weight * cat.inr_per_tonne_min / 1000, 2)
        scrap_max  = round(weight * cat.inr_per_tonne_max / 1000, 2)
        inr_rate   = TOKEN_INR_RATES.get(cat.code, 1.5)
        token_inr  = round(weight * inr_rate * 0.5, 2)
        carbon_inr = round(weight * 1.0, 2)
        total_inr  = round(scrap_min + token_inr + carbon_inr, 2)

        boxes.append(DetectionBox(
            category_code=cat.code,
            category_name=name,
            confidence=round(cls_r.get("confidence", 0.5), 3),
            recyclable=cat.recyclable,
            severity_score=cat.severity_score,
            sub_type_route=cat.sub_type_route,
            weight_kg_estimate=weight,
            co2e_avoided_kg=co2e,
            bbox_xyxy=det.get("bbox", [0, 0, 0, 0]),
            scrap_value_inr_min=scrap_min,
            scrap_value_inr_max=scrap_max,
            token_value_inr=token_inr,
            carbon_credit_inr=carbon_inr,
            total_value_inr_estimate=total_inr,
        ))
    logger.info("Enrich: %d detections → %d boxes (dropped %d)",
                len(detections), len(boxes), len(detections) - len(boxes))
    return boxes


@app.post("/classify", response_model=ClassificationResult)
async def classify_waste(
    image: UploadFile = File(...),
    work_order_id: Optional[str] = None,
    location_lat: Optional[float] = None,
    location_lng: Optional[float] = None,
    model: ClassificationModel = Depends(get_model),
):
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "Image must be JPEG, PNG or WebP")
    raw = await image.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 20MB")
    img_hash = hashlib.sha256(raw).hexdigest()[:16]
    logger.info("classify: hash=%s len=%d prefix=%s", img_hash, len(raw), raw[:50].hex())
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        logger.error("PIL open failed: %s | data=%s", e, raw[:100])
        raise

    try:
        detections      = model.detect(img)
        classifications  = model.classify(img, detections)
        volume_est      = model.estimate_volume(img, detections)
        ai_narrative    = model.generate_narrative(img, detections)
    except Exception as e:
        logger.error("Inference pipeline failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Inference error: {e}")

    boxes = _enrich(detections, classifications, volume_est)
    
    logger.info("RESULT: %d boxes. total_detections was %d", len(boxes), len(detections))
    for i, b in enumerate(boxes):
        logger.info("BOX [%d]: code=%s weight=%f", i, b.category_code, b.weight_kg_estimate)

    # BUG-04 FIX: dominant by confidence × severity, W16 penalised
    def _score(b: DetectionBox) -> float:
        return b.confidence * _SEV.get(b.category_code, 5) * (
            0.6 if b.category_code == "W16" else 1.0)
    dominant = max(boxes, key=_score) if boxes else None

    total_weight    = round(sum(b.weight_kg_estimate for b in boxes), 3)
    total_co2e      = round(sum(b.co2e_avoided_kg   for b in boxes), 4)
    total_scrap_min = round(sum(b.scrap_value_inr_min for b in boxes), 2)
    total_scrap_max = round(sum(b.scrap_value_inr_max for b in boxes), 2)
    total_token_inr = round(sum(b.token_value_inr     for b in boxes), 2)
    total_carbon_inr= round(sum(b.carbon_credit_inr   for b in boxes), 2)
    grand_total_inr = round((total_scrap_min + total_scrap_max) / 2 + total_token_inr + total_carbon_inr, 2)
    # 3. No grouping - show all individual detections for high-density visibility
    for det in boxes:
        det.count = 1
        det.weight_kg_estimate = round(det.weight_kg_estimate, 4)
    
    final_boxes = boxes
    cats_found = list({b.category_code for b in boxes})
    
    logger.info("FINAL RESULT: weight=%f co2=%f", total_weight, total_co2e)

    return ClassificationResult(
        request_id=img_hash,
        work_order_id=work_order_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        image_hash=img_hash,
        location_lat=location_lat,
        location_lng=location_lng,
        detections=final_boxes,
        total_detections=len(boxes),
        dominant_category=dominant.category_code if dominant else "W16",
        dominant_category_name=dominant.category_name if dominant else "Mixed Waste",
        severity="High Impact" if (dominant and dominant.severity_score >= 7) else "Medium Impact" if (dominant and dominant.severity_score >= 4) else "Low Impact",
        total_weight_kg_estimate=total_weight,
        total_co2e_avoided_kg=total_co2e,
        categories_found=cats_found,
        recyclable_fraction=round(
            sum(1 for b in boxes if b.recyclable) / max(len(boxes), 1), 2),
        volume_estimate=volume_est,
        ai_narrative=ai_narrative,
        total_scrap_value_inr_min=total_scrap_min,
        total_scrap_value_inr_max=total_scrap_max,
        total_token_value_inr=total_token_inr,
        total_carbon_credit_inr=total_carbon_inr,
        grand_total_value_inr=grand_total_inr,
    )


@app.post("/classify/debug")
async def classify_debug(
    image: UploadFile = File(...),
    model: ClassificationModel = Depends(get_model),
):
    """
    Debug endpoint — returns every intermediate step.
    Aditya: call this to see exactly where the pipeline breaks.
    """
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    step = {}
    step["0_model_state"] = {
        "stub_mode": model._stub_mode,
        "device": model.device,
        "yolo_loaded": model._yolo is not None,
        "efficientnet_loaded": model._efficientnet is not None,
        "midas_loaded": model._midas is not None,
    }
    step["1_image"] = {"width": img.size[0], "height": img.size[1]}

    try:
        detections = model.detect(img)
        step["2_detections"] = detections
        step["2_detection_count"] = len(detections)
    except Exception as e:
        step["2_error"] = str(e)
        return step

    try:
        classifications = model.classify(img, detections)
        step["3_classifications"] = classifications
    except Exception as e:
        step["3_error"] = str(e)
        return step

    try:
        vol = model.estimate_volume(img, detections)
        step["4_volume"] = {"method": vol.method, "total_m3": vol.total_volume_m3}
    except Exception as e:
        step["4_error"] = str(e)
        return step

    boxes = _enrich(detections, classifications, vol)
    step["5_boxes_after_enrich"] = len(boxes)
    step["5_boxes_detail"] = [
        {"code": b.category_code, "name": b.category_name,
         "conf": b.confidence, "weight_kg": b.weight_kg_estimate}
        for b in boxes
    ]

    # Check each detection for taxonomy lookup failure
    enrichment_trace = []
    for det, cls_r in zip(detections, classifications):
        code = cls_r.get("category_code", "MISSING_KEY")
        cat = get_by_code(code)
        enrichment_trace.append({
            "detected_code":   det.get("category_code"),
            "classified_code": code,
            "taxonomy_found":  cat is not None,
            "category_name":   cat.name if cat else None,
        })
    step["6_enrichment_trace"] = enrichment_trace

    return step


@app.get("/categories")
def list_categories():
    return {
        "total": len(INDICWASTE),
        "categories": [
            {"code": c.code, "name": c.name, "recyclable": c.recyclable,
             "severity_score": c.severity_score, "sub_type_route": c.sub_type_route,
             "co2e_avoided_per_tonne": c.co2e_avoided_per_tonne,
             "inr_range": [c.inr_per_tonne_min, c.inr_per_tonne_max]}
            for c in INDICWASTE.values()
        ],
    }


class BatchRequest(BaseModel):
    images_base64: list[str] = Field(..., max_items=10)
    work_order_id: Optional[str] = None

@app.post("/classify/batch")
async def classify_batch(req: BatchRequest,
                         model: ClassificationModel = Depends(get_model)):
    results = []
    for b64 in req.images_base64:
        try:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            dets = model.detect(img)
            cls_ = model.classify(img, dets)
            vol  = model.estimate_volume(img, dets)
            boxes = _enrich(dets, cls_, vol)
            results.append({"status": "ok", "total_detections": len(boxes)})
        except Exception as e:
            results.append({"status": "error", "detail": str(e)})
    return {"results": results, "work_order_id": req.work_order_id}
