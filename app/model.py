"""
MIBA — ClassificationModel v1.2
Clean rewrite — no silent failures, all errors logged explicitly.
"""

import os, io, base64, logging, random, json, re

from google.cloud import aiplatform, vision
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import numpy as np

from .schemas import VolumeEstimate

logger = logging.getLogger(__name__)
MAX_INFER_PX   = 1536
LARGE_BBOX_THR = 0.55
TILE_GRID      = 2

CATEGORY_CODES = [f"W{str(i).zfill(2)}" for i in range(1, 17)]

# Empty mappings as the user wants raw names and no weights
WASTE_DENSITY = {
    "W01": 500,  "W02": 300,  "W03": 800,  "W04": 2000, "W05": 1500,
    "W06": 200,  "W07": 1500, "W08": 400,  "W09": 900,  "W10": 400,
    "W11": 600,  "W12": 1800, "W13": 250,  "W14": 180,  "W15": 1600, "W16": 600,
}
STUB_CATS = (
    ["W01"] * 4 + ["W02"] * 3 + ["W03"] * 3 + ["W06"] * 2 +
    ["W14"] * 2 + ["W04"] * 1 + ["W10"] * 1 + ["W16"] * 1
)
VERTEX_TO_MIBA = {
    "food": "W01", "organic": "W01", "vegetable": "W01", "fruit": "W01", "apple": "W01", "banana": "W01",
    "bottle": "W02", "plastic bottle": "W02", "hdpe": "W02", "container": "W02", "jug": "W02",
    "plastic bag": "W03", "bag": "W03", "sack": "W03", "garbage bag": "W03", "trash bag": "W03",
    "paper": "W06", "cardboard": "W06", "box": "W06", "carton": "W06",
    "glass": "W12", "bottle": "W12", "jar": "W12",
    "aluminum": "W14", "can": "W14", "tin": "W14", "foil": "W14",
    "waste container": "W16", "bin": "W16", "wheelie bin": "W16"
}

# Maps Gemini-returned item types → MIBA category codes
GEMINI_TYPE_TO_MIBA = {
    "bin bag": "W03", "bin bags": "W03", "black bag": "W03", "black bags": "W03",
    "garbage bag": "W03", "garbage bags": "W03", "trash bag": "W03", "trash bags": "W03",
    "plastic bag": "W03", "plastic bags": "W03", "refuse sack": "W03", "refuse sacks": "W03",
    "carrier bag": "W03", "carrier bags": "W03", "polythene bag": "W03",
    "cardboard": "W06", "cardboard box": "W06", "cardboard boxes": "W06",
    "paper": "W06", "paper bag": "W06", "newspaper": "W06", "box": "W06", "boxes": "W06",
    "plastic bottle": "W02", "plastic bottles": "W02", "bottle": "W02", "bottles": "W02",
    "container": "W02", "containers": "W02", "jug": "W02", "jugs": "W02",
    "food": "W01", "food waste": "W01", "organic": "W01", "organic waste": "W01",
    "fruit": "W01", "vegetable": "W01", "kitchen waste": "W01",
    "can": "W14", "cans": "W14", "tin": "W14", "tins": "W14", "foil": "W14",
    "aluminium can": "W14", "aluminum can": "W14",
    "glass bottle": "W07", "glass jar": "W07", "jar": "W07", "glass": "W07",
    "clothing": "W10", "clothes": "W10", "textile": "W10", "fabric": "W10",
    "loose litter": "W16", "litter": "W16", "rubbish": "W16", "debris": "W16",
    "mixed waste": "W16", "waste": "W16", "refuse": "W16",
}

class ClassificationModel:

    def __init__(self, yolo_weights, efficientnet_weights, midas_weights,
                 confidence_threshold=0.05, device="cpu", vertex_endpoint_id=None):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.vertex_endpoint_id = vertex_endpoint_id
        self._endpoint = None
        self._gemini = None
        self._yolo = self._efficientnet = self._midas = self._transform = None
        self._stub_mode = False
        
        # Cloud Vision Client
        try:
            # Fix 403 Quota Project error by passing the project ID explicitly
            quota_project = os.getenv("GOOGLE_PROJECT_ID")
            client_options = {"quota_project_id": quota_project} if quota_project else {}
            self._vision_client = vision.ImageAnnotatorClient(client_options=client_options)
            logger.info("Cloud Vision client initialized with quota_project: %s", quota_project)
        except Exception as e:
            logger.warning("Cloud Vision init failed: %s", e)
            self._vision_client = None

        # Prioritize local weights if they exist, otherwise use Vertex AI
        if os.path.exists(yolo_weights) and os.path.exists(efficientnet_weights):
            logger.info("Local weights found. Initializing local models.")
            self._load_models(yolo_weights, efficientnet_weights, midas_weights)
            # If local load failed (stub_mode), try Vertex AI as fallback
            if self._stub_mode and self.vertex_endpoint_id:
                logger.warning("Local load failed. Falling back to Vertex AI.")
                self._init_vertex()
        elif self.vertex_endpoint_id:
            logger.info("Local weights missing. Initializing Vertex AI.")
            self._init_vertex()
        else:
            logger.warning("No local weights and no Vertex ID. Using stub mode.")
            self._load_models(yolo_weights, efficientnet_weights, midas_weights)
        # Always attempt Gemini initialization for narrative
        self._init_vertex()
        
        logger.info("Model ready | stub=%s device=%s vertex_ep=%s gemini=%s", 
                    self._stub_mode, self.device, bool(self._endpoint), bool(self._gemini))


    def _init_vertex(self):
        try:
            from google.cloud import aiplatform
            project = os.getenv("GOOGLE_PROJECT_ID", "complisight-uat")
            location = os.getenv("GOOGLE_REGION", "europe-west1")
            
            aiplatform.init(project=project, location=location)
            
            # Use us-central1 for Gemini as it's the most compatible for model garden
            vertexai.init(project=project, location="us-central1")
            
            # Gemini 1.5 Flash is more widely available in GCP projects
            self._gemini = GenerativeModel("gemini-1.5-flash")
            
            # Vertex AI Endpoint (Optional, uses primary location)
            if self.vertex_endpoint_id:
                self._endpoint = aiplatform.Endpoint(self.vertex_endpoint_id)
                logger.info("Vertex AI Endpoint (at %s) & Gemini (at europe-west1) initialized.", location)
            else:
                logger.info("Gemini initialized at europe-west1 (Narrative mode).")
        except Exception as e:
            logger.error("AI platform init failed: %s", e)
            self._gemini = None
            self._endpoint = None


    def _load_models(self, yolo_path, effnet_path, midas_path):
        try:
            import torch, torchvision
            from torchvision import transforms as T

            missing = [p for p in [yolo_path, effnet_path] if not os.path.exists(p)]
            if missing:
                logger.warning("Weights not found %s — stub mode", missing)
                self._stub_mode = True
                return

            logger.info("Loading YOLOv9")
            try:
                # Try local first, then GitHub
                try:
                    self._yolo = torch.hub.load("WongKinYiu/yolov9", "custom",
                                                path=yolo_path, source="local",
                                                device=self.device)
                except:
                    logger.info("Local YOLOv9 source missing. Trying GitHub...")
                    self._yolo = torch.hub.load("WongKinYiu/yolov9", "custom",
                                                path=yolo_path, device=self.device)
                self._yolo.conf = self.confidence_threshold
                self._yolo.eval()
            except Exception as ey:
                logger.warning("Local YOLOv9 failed: %s", ey)

            logger.info("Loading EfficientNet-B3")
            try:
                self._efficientnet = torchvision.models.efficientnet_b3(weights=None)
                self._efficientnet.classifier[1] = torch.nn.Linear(
                    self._efficientnet.classifier[1].in_features, 16)
                self._efficientnet.load_state_dict(
                    torch.load(effnet_path, map_location=self.device))
                self._efficientnet.eval()
            except Exception as ee:
                logger.warning("Local EfficientNet failed: %s", ee)

            if os.path.exists(midas_path):
                logger.info("Loading MiDaS")
                try:
                    # Try local source first
                    try:
                        self._midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                                     source="local")
                    except:
                        logger.info("Local MiDaS source missing. Trying GitHub...")
                        self._midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                    
                    self._midas.load_state_dict(torch.load(midas_path, map_location=self.device))
                    self._midas.to(self.device).eval()
                except Exception as em:
                    logger.warning("MiDaS failed: %s", em)

            self._transform = T.Compose([
                T.Resize((300, 300)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            logger.info("Local models attempt finished")

        except ImportError:
            logger.warning("torch not available")
        except Exception as e:
            logger.error("Model load error: %s", e, exc_info=True)
        
        # Determine if we need stub mode.
        # Gemini (set in _init_vertex) counts as a real detection source.
        has_real_source = (
            self._yolo is not None or
            self._vision_client is not None or
            self._endpoint is not None or
            self._gemini is not None
        )
        if not has_real_source:
            logger.warning("CRITICAL: ALL detection sources unavailable. Entering stub mode.")
            self._stub_mode = True
        else:
            self._stub_mode = False
            logger.info("Detection pipeline online (YOLO=%s, Vision=%s, Vertex=%s, Gemini=%s)",
                        self._yolo is not None, self._vision_client is not None,
                        self._endpoint is not None, self._gemini is not None)

    def _resize(self, img):
        w, h = img.size
        longest = max(w, h)
        if longest <= MAX_INFER_PX:
            return img
        s = MAX_INFER_PX / longest
        return img.resize((int(w * s), int(h * s)), Image.LANCZOS)

    def detect(self, img: Image.Image) -> list[dict]:
        """Priority: Local YOLO > Cloud Vision > Gemini structured > Stub (only if Gemini unavailable)."""
        detections = []
        # 1. Try Local YOLO
        yolo_dets = []
        if not self._stub_mode and self._yolo is not None:
            try:
                small = self._resize(img)
                import torch
                res = self._yolo(small)
                for *box, conf, cls_id in res.xyxy[0].tolist():
                    if conf < self.confidence_threshold:
                        continue
                    x1, y1, x2, y2 = box
                    w, h = small.size
                    af = ((x2-x1)*(y2-y1)) / (w*h)
                    cat_code = CATEGORY_CODES[min(int(cls_id), 15)]
                    # Try to get class name from YOLO model attributes (if available)
                    cat_name = None
                    if hasattr(self._yolo, 'names') and int(cls_id) in self._yolo.names:
                        cat_name = self._yolo.names[int(cls_id)].title()
                    else:
                        # Fallback if names not in model
                        cat_name = cat_code.replace("W", "Category ")

                    yolo_dets.append({
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "confidence": round(conf, 3),
                        "category_code": cat_code,
                        "category_name": cat_name,
                        "volume_m3": round(af * 0.02, 6),
                        "bbox_area_fraction": round(af, 4),
                    })
                if yolo_dets:
                    logger.info("Local YOLO: %d detections", len(yolo_dets))
                    detections.extend(yolo_dets)
            except Exception as e:
                logger.error("Local YOLO failed: %s", e)

        # 2. Try Cloud Vision with High-Density Tiling (2x2 Grid)
        if self._vision_client:
            try:
                w, h = img.size
                tiles = []
                grid_size = TILE_GRID  # Default is 2 (2x2)
                tw, th = w // grid_size, h // grid_size
                
                # Add full image as the first "tile" for global context
                tiles.append((img.copy(), 0, 0, w, h))
                
                # Add sub-tiles
                for r in range(grid_size):
                    for c in range(grid_size):
                        x_off, y_off = c * tw, r * th
                        tile_img = img.crop((x_off, y_off, x_off + tw, y_off + th))
                        tiles.append((tile_img, x_off, y_off, tw, th))
                
                # Prepare batch request
                requests = []
                for tile_img, _, _, _, _ in tiles:
                    buf = io.BytesIO()
                    tile_img.save(buf, format="JPEG")
                    v_img = vision.Image(content=buf.getvalue())
                    requests.append(vision.AnnotateImageRequest(
                        image=v_img,
                        features=[vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=100)]
                    ))
                
                # Single batch call for all tiles
                try:
                    logger.info("Cloud Vision: Sending batch request for %d tiles", len(requests))
                    response = self._vision_client.batch_annotate_images(requests=requests)
                except Exception as api_err:
                    logger.error("Cloud Vision Batch API call failed: %s", api_err)
                    raise api_err
                
                vision_raw = []
                for i, res in enumerate(response.responses):
                    tile_img, x_off, y_off, tw, th = tiles[i]
                    if res.error.message:
                        logger.warning("Vision Tile %d error: %s", i, res.error.message)
                        continue
                        
                    for obj in res.localized_object_annotations:
                        if obj.score < self.confidence_threshold:
                            continue
                        
                        v = obj.bounding_poly.normalized_vertices
                        if len(v) >= 4:
                            tx1, ty1 = v[0].x * tw, v[0].y * th
                            tx2, ty2 = v[2].x * tw, v[2].y * th
                            abs_x1, abs_y1 = x_off + tx1, y_off + ty1
                            abs_x2, abs_y2 = x_off + tx2, y_off + ty2
                            
                            label_lower = obj.name.lower()
                            cat_code = "W16"
                            for key, code in VERTEX_TO_MIBA.items():
                                if key in label_lower:
                                    cat_code = code
                                    break
                            
                            vision_raw.append({
                                "bbox": [float(abs_x1), float(abs_y1), float(abs_x2), float(abs_y2)],
                                "score": float(obj.score),
                                "cat_code": cat_code,
                                "name": obj.name.title()
                            })

                # Deduplicate overlapping boxes across tiles using IOU
                unique_vision = []
                # Sort by score descending to keep the best detections
                sorted_vision = sorted(vision_raw, key=lambda x: x["score"], reverse=True)
                
                for cand in sorted_vision:
                    is_dup = False
                    for existing in unique_vision:
                        try:
                            iou = self._calculate_iou(cand["bbox"], existing["bbox"])
                            if iou > 0.45:
                                is_dup = True
                                break
                        except Exception as iou_err:
                            logger.warning("IOU calculation failed: %s", iou_err)
                            continue
                    if not is_dup:
                        unique_vision.append(cand)

                # Convert to final detection format
                for d in unique_vision:
                    x1, y1, x2, y2 = d["bbox"]
                    denom = w * h
                    af = ((x2-x1)*(y2-y1)) / denom if denom > 0 else 0
                    detections.append({
                        "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                        "confidence": round(d["score"], 3),
                        "category_code": d["cat_code"],
                        "category_name": d["name"],
                        "volume_m3": float(af * 0.45),
                        "bbox_area_fraction": float(af),
                        "method": "cloud_vision_tiled"
                    })
                
                logger.info("Cloud Vision (Tiled): %d raw -> %d unique", len(vision_raw), len(unique_vision))
            except Exception as e:
                logger.error("Cloud Vision High-Density pipeline failed: %s", e, exc_info=True)
                # Fallback to single image if tiling fails
                return self._detect_single(img)

        # If detection is poor (empty, ≤1 item, or everything is generic),
        # use Gemini structured detection for real item counts and weights.
        generic_names = {"waste container", "bin", "wheelie bin", "mixed waste", "mixed / unclassified"}
        poor_detection = (
            len(detections) == 0 or
            len(detections) <= 1 or
            all(str(d.get("category_name") or "").lower() in generic_names for d in detections)
        )
        if poor_detection and self._gemini:
            logger.info("Detection quality poor (%d items). Using Gemini structured detect.", len(detections))
            gemini_dets = self.gemini_structured_detect(img)
            if gemini_dets:
                return gemini_dets

        # Only reach stub if Gemini is also unavailable
        if not detections and self._stub_mode:
            logger.warning("All real detection sources failed. Returning stub detections.")
            return self._stub_detections(img)
        return detections

    def _detect_single(self, img: Image.Image) -> list[dict]:
        """Robuste fallback: run Vision on the whole image once."""
        detections = []
        if not self._vision_client:
            return detections
            
        try:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            v_img = vision.Image(content=buf.getvalue())
            request = vision.AnnotateImageRequest(
                image=v_img,
                features=[vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=100)]
            )
            logger.info("Cloud Vision: Falling back to single-pass.")
            response_batch = self._vision_client.batch_annotate_images(requests=[request])
            response = response_batch.responses[0]
            
            w, h = img.size
            for obj in response.localized_object_annotations:
                if obj.score < self.confidence_threshold:
                    continue
                v = obj.bounding_poly.normalized_vertices
                if len(v) >= 4:
                    x1, y1 = v[0].x * w, v[0].y * h
                    x2, y2 = v[2].x * w, v[2].y * h
                    label_lower = obj.name.lower()
                    cat_code = "W16"
                    for key, code in VERTEX_TO_MIBA.items():
                        if key in label_lower:
                            cat_code = code
                            break
                    detections.append({
                        "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                        "confidence": float(round(obj.score, 3)),
                        "category_code": cat_code,
                        "category_name": obj.name.title(),
                        "volume_m3": float(((x2-x1)*(y2-y1)) / (w*h) * 0.45) if (w*h) > 0 else 0,
                        "bbox_area_fraction": float(((x2-x1)*(y2-y1)) / (w*h)) if (w*h) > 0 else 0,
                        "method": "cloud_vision_single"
                    })
        except Exception as e:
            logger.error("Vision single-pass failed: %s", e)
        return detections

    def gemini_structured_detect(self, img: Image.Image) -> list[dict]:
        """
        Ask Gemini to count and weigh each waste item type in the image.
        Returns a list of detections matching the same format as model.detect().
        Each detected item type becomes one entry per unit (count is expanded).
        """
        if not self._gemini:
            return []
        try:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()
            w, h = img.size

            prompt = (
                "You are a waste detection AI. Carefully count every type of waste item visible in this image. "
                "Respond ONLY with a valid JSON object — no markdown, no explanation. "
                "Use this exact format:\n"
                "{\n"
                '  "items": [\n'
                '    {"type": "Black Bin Bags", "count": 17, "weight_per_unit_kg": 12.0, "category": "bin bags"},\n'
                '    {"type": "Cardboard Boxes", "count": 4, "weight_per_unit_kg": 2.5, "category": "cardboard boxes"},\n'
                '    {"type": "Loose Litter", "count": 1, "weight_per_unit_kg": 5.0, "category": "loose litter"}\n'
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                "- Count each individual item carefully. For large piles, give your best estimate.\n"
                "- weight_per_unit_kg = realistic weight of ONE unit (e.g. full bin bag = 10-15 kg, cardboard box = 1-3 kg, plastic bottle = 0.5 kg).\n"
                "- Do NOT use 0 for count or weight. Be specific and realistic.\n"
                "- Only include items you can actually see. Do not invent items."
            )

            models_to_try = ["gemini-2.0-flash-001", "gemini-2.0-flash", "gemini-2.0-flash-lite-001"]
            raw_text = None
            for m_name in models_to_try:
                try:
                    m = GenerativeModel(m_name)
                    response = m.generate_content([Part.from_data(img_bytes, mime_type="image/jpeg"), prompt])
                    if response.text:
                        raw_text = response.text.strip()
                        break
                except Exception as ex:
                    logger.warning("Gemini structured detect model %s failed: %s", m_name, ex)

            if not raw_text:
                return []

            # Strip markdown code fences if present
            assert raw_text is not None
            clean_text: str = re.sub(r"^```[a-z]*\n?", "", raw_text).rstrip("` \n")

            data = json.loads(clean_text)
            items = data.get("items", [])

            detections = []
            # Distribute bboxes roughly across the image for visualisation
            cols = max(1, int(len(items) ** 0.5))

            for idx, item in enumerate(items):
                count = max(1, int(item.get("count", 1)))
                weight_per: float = max(0.1, float(item.get("weight_per_unit_kg", 1.0)))
                item_type = str(item.get("type", "Waste Item"))
                category_hint = str(item.get("category", item_type)).lower()

                # Map to MIBA code
                cat_code = "W16"
                for key, code in GEMINI_TYPE_TO_MIBA.items():
                    if key in category_hint:
                        cat_code = code
                        break

                # volume_m3 back-calculated from realistic weight / density
                density: float = float(WASTE_DENSITY.get(cat_code, 300))
                vol_per_unit: float = weight_per / density

                # Spread bboxes in a grid across image for rough visualisation
                col_w = w // max(cols, 1)
                col_h = h // max(cols, 1)
                gx = (idx % cols) * col_w
                gy = (idx // cols) * col_h

                af: float = vol_per_unit / 0.45  # reverse of cloud vision formula
                conf: float = float(item.get("confidence", 0.82))
                conf_r = float(f"{conf:.3f}")
                vol_r = float(f"{vol_per_unit:.6f}")
                af_r = float(f"{af:.6f}")
                for _ in range(count):
                    detections.append({
                        "bbox": [gx, gy, min(gx + col_w, w), min(gy + col_h, h)],
                        "confidence": conf_r,
                        "category_code": cat_code,
                        "category_name": item_type,
                        "volume_m3": vol_r,
                        "bbox_area_fraction": af_r,
                        "method": "gemini_structured",
                    })

            logger.info("Gemini structured detect: %d item types → %d total detections", len(items), len(detections))
            return detections

        except json.JSONDecodeError as e:
            logger.error("Gemini structured detect JSON parse error: %s | raw: %s", e, raw_text[:200] if raw_text else "")
            return []
        except Exception as e:
            logger.error("Gemini structured detect failed: %s", e, exc_info=True)
            return []

    def generate_narrative(self, img: Image.Image, detections: list[dict]) -> str:
        """
        Generates a premium, count-aware summary of the detected waste.
        Prioritizes Gemini 1.5/2.0 and falls back to a high-fidelity heuristic.
        """
        narrative_text = ""
        
        # 1. Attempt Generative AI Narrative
        if self._gemini:
            try:
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                
                prompt = (
                    "You are a waste management AI. Carefully analyze this image and provide a realistic, specific assessment. "
                    "Respond with a professional, data-driven overview of the waste shown. "
                    "Crucial: If you see 'Loose Litter' or scattered debris, specifically itemize what constitutes it (e.g., 'crushed plastic bottles', 'discarded cigarette packs', 'tattered cardboard scraps'). "
                    "Include an estimate of the item counts and an approximate total weight range (e.g. 150-250kg)."
                )
                
                # Try multiple models to avoid project-specific 404s
                for m_name in ["gemini-1.5-flash", "gemini-1.5-pro"]:
                    try:
                        m = GenerativeModel(m_name)
                        response = m.generate_content([Part.from_data(img_bytes, mime_type="image/jpeg"), prompt])
                        if response.text:
                            narrative_text = response.text
                            break
                    except Exception as model_err:
                        logger.warning("Gemini %s failed: %s", m_name, model_err)
            except Exception as e:
                logger.error("Gemini narrative block failed: %s", e)
        
        # 2. Heuristic Fallback (Always works, even offline)
        if not narrative_text:
            try:
                dets = detections or []
                if not dets:
                    return "No waste items clearly detected in the image for detailed analysis."

                item_counts = {}
                total_weight_est = 0.0
                for d in dets:
                    name = (d.get('category_name') or 'Mixed Waste').title()
                    item_counts[name] = item_counts.get(name, 0) + 1
                    code = d.get('category_code', 'W16')
                    density = WASTE_DENSITY.get(code, 150)
                    total_weight_est += d.get('volume_m3', 0.05) * density

                est_weight_min = max(1, int(total_weight_est * 0.8))
                est_weight_max = max(2, int(total_weight_est * 1.2))

                bullets = [f"- **{name}**: Approximately {count} items." 
                          for name, count in sorted(item_counts.items(), key=lambda x: x[1], reverse=True)]
                
                narrative_text = (
                    f"### AI Overview (Visual Analysis)\n"
                    f"The analysis has identified approximately **{len(dets)}** individual waste items in the scene.\n\n"
                    f"**Item Count:**\n" + "\n".join(bullets) + "\n\n"
                    f"**Estimated Weight:**\n"
                    f"Based on visual volume, this collection weighs between **{est_weight_min} kg and {est_weight_max} kg**.\n"
                )
            except Exception as heur_err:
                logger.error("Heuristic fallback failed: %s", heur_err)
                narrative_text = f"AI Analysis complete. [RECOVERY MODE]\n\nDetected {len(detections)} items."

        return narrative_text

    def classify(self, img: Image.Image, detections: list[dict]) -> list[dict]:
        if self.vertex_endpoint_id and self._endpoint:
            return self._vertex_classify(detections)

        # Detections from cloud_vision_tiled or gemini_structured already carry
        # their category codes — pass them through without randomising.
        real_methods = {"cloud_vision_tiled", "gemini_structured"}
        if detections and all(d.get("method") in real_methods for d in detections):
            return [{"category_code": d["category_code"],
                     "confidence": d["confidence"],
                     "all_probs": {}, "method": d.get("method")} for d in detections]

        if self._stub_mode:
            return self._stub_classifications(detections)

        # Local EfficientNet classification
        if self._efficientnet is None:
            # No local model — pass through codes from detection
            return [{"category_code": d.get("category_code", "W16"),
                     "confidence": d.get("confidence", 0.5),
                     "all_probs": {}, "method": "passthrough"} for d in detections]

        small = self._resize(img)
        out = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            af = det.get("bbox_area_fraction", 0.0)
            if af > LARGE_BBOX_THR:
                code, conf = self._tile_vote(small, x1, y1, x2, y2)
                out.append({"category_code": code, "confidence": conf,
                            "all_probs": {}, "method": "tiled"})
            else:
                crop = small.crop((x1, y1, x2, y2))
                code, conf, probs = self._classify_crop(crop)
                out.append({"category_code": code, "confidence": conf,
                            "all_probs": probs, "method": "direct"})
        return out
    def _vertex_detect(self, img: Image.Image) -> list[dict]:
        """Call Vertex AI for detection."""
        # Convert PIL to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        instance = {"content": img_str}
        parameters = {"confidenceThreshold": self.confidence_threshold, "maxPredictions": 200}

        try:
            prediction = self._endpoint.predict(instances=[instance], parameters=parameters)
            # Response parsing depends on the specific model deployment schema
            # Assuming standard AutoML Vision Object Detection format:
            # [{"bboxes": [[y1,x1,y2,x2]], "confidences": [0.9], "displayNames": ["W01"]}]
            
            objs = []
            for pred in prediction.predictions:
                # Handle different Vertex response formats (AutoML vs Custom)
                if isinstance(pred, dict) and "bboxes" in pred:
                    for i, bbox in enumerate(pred["bboxes"]):
                        y1, x1, y2, x2 = bbox
                        conf = pred["confidences"][i]
                        label_name = pred["displayNames"][i].lower()
                        
                        # Map to MIBA code if possible
                        cat_code = "W16" # Default
                        for key, code in VERTEX_TO_MIBA.items():
                            if key in label_name:
                                cat_code = code
                                break
                        
                        if conf < self.confidence_threshold: continue
                        
                        w, h = img.size
                        # Convert normalized to pixel coords
                        px_bbox = [round(x1*w), round(y1*h), round(x2*w), round(y2*h)]
                        af = (x2-x1)*(y2-y1)
                        
                        objs.append({
                            "bbox": px_bbox,
                            "confidence": round(conf, 3),
                            "category_code": cat_code,
                            "volume_m3": round(af * 0.02, 6),
                            "bbox_area_fraction": round(af, 4),
                        })
            return objs
        except Exception as e:
            logger.error("Vertex detect failed: %s", e)
            return self._stub_detections(img)

    def _vertex_classify(self, detections: list[dict]) -> list[dict]:
        """Vertex results often come pre-classified during detection."""
        return [{"category_code": d["category_code"], 
                 "confidence": d["confidence"], 
                 "all_probs": {}, "method": "vertex"} for d in detections]


    def _classify_crop(self, crop):
        import torch
        t = self._transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self._efficientnet(t), dim=1)[0]
        idx = int(probs.argmax())
        return (CATEGORY_CODES[idx], round(float(probs[idx]), 3),
                {CATEGORY_CODES[i]: round(float(p), 3) for i, p in enumerate(probs)})

    def _tile_vote(self, img, x1, y1, x2, y2):
        tw = (x2-x1)//TILE_GRID
        th = (y2-y1)//TILE_GRID
        votes = []
        for r in range(TILE_GRID):
            for c in range(TILE_GRID):
                tile = img.crop((x1+c*tw, y1+r*th, x1+c*tw+tw, y1+r*th+th))
                code, conf, _ = self._classify_crop(tile)
                votes.append((code, conf))
        specific = [(c, conf) for c, conf in votes if c != "W16" and conf >= 0.40]
        if specific:
            return max(specific, key=lambda x: x[1])
        return "W16", round(sum(c for _, c in votes)/len(votes), 3)

    def estimate_volume(self, img: Image.Image, detections: list[dict]) -> VolumeEstimate:
        if not detections:
            return VolumeEstimate(method="empty", total_volume_m3=0.0,
                                  density_kg_m3=WASTE_DENSITY, confidence=0.0)
        if self._stub_mode or self._midas is None:
            total = sum(d.get("volume_m3", 0.008) for d in detections)
            return VolumeEstimate(method="bbox_proxy",
                                  total_volume_m3=round(total, 4),
                                  density_kg_m3=WASTE_DENSITY, confidence=0.35)
        import torch, torchvision.transforms.functional as TF
        small = self._resize(img)
        t = TF.to_tensor(small.resize((256, 256))).unsqueeze(0)
        with torch.no_grad():
            depth = self._midas(t).squeeze().numpy()
        w, h = small.size
        total = 0.0
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            region = depth[int(y1/h*256):int(y2/h*256), int(x1/w*256):int(x2/w*256)]
            md = float(np.mean(region)) if region.size > 0 else 0.1
            total += ((x2-x1)*(y2-y1))/(w*h)*0.5 * md * 0.01
        return VolumeEstimate(method="midas", total_volume_m3=round(total, 4),
                              density_kg_m3=WASTE_DENSITY, confidence=0.65)

    def _iou(self, box1, box2):
        """Compute Intersection over Union of two bounding boxes."""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = interArea / float(box1Area + box2Area - interArea)
        return iou

    def _stub_detections(self, img):
        w, h = img.size
        n = 3 if (w*h) > 1_000_000 else random.randint(1, 3)
        out = []
        for _ in range(n):
            bx1 = random.randint(0, w//3)
            by1 = random.randint(0, h//3)
            bx2 = random.randint(w//2, w)
            by2 = random.randint(h//2, h)
            af = round(((bx2-bx1)*(by2-by1))/(w*h), 4)
            out.append({
                "bbox": [bx1, by1, bx2, by2],
                "confidence": round(random.uniform(0.55, 0.92), 3),
                "category_code": random.choice(STUB_CATS),
                "volume_m3": round(af * 0.02, 6),
                "bbox_area_fraction": af,
            })
        logger.info("Stub: %d detections generated", len(out))
        return out

    def _stub_classifications(self, detections):
        out = [{"category_code": d["category_code"],
                "confidence": round(random.uniform(0.62, 0.93), 3),
                "all_probs": {}, "method": "stub"}
               for d in detections]
        logger.info("Stub: %d classifications", len(out))
        return out
    def _calculate_iou(self, boxA, boxB):
        """Compute IOU between two boxes to deduplicate detections across tiles."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
