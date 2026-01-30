"""
FastAPI API untuk GNN Road Criticality dengan CNN Integration.

Run dengan:
    uv run uvicorn src.api:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from pydantic import BaseModel, Field
import tempfile
import shutil
from datetime import datetime
from enum import Enum
import os
import httpx

import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

from .impact_propagation import RoadDamageAnalyzer
from .road_state_manager import get_state_manager, RoadStateManager
from .unified_pipeline import UnifiedRoadAnalyzer


# === Base directory resolution ===
BASE_DIR = Path(__file__).parent.parent

# Load environment variables from .env file (explicit path)
load_dotenv(BASE_DIR / ".env")


# === Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load analyzer, state manager, and unified pipeline at startup."""
    print("Loading RoadDamageAnalyzer...")
    app.state.analyzer = RoadDamageAnalyzer(
        model_dir=BASE_DIR / "models",
        data_dir=BASE_DIR / "data"
    )
    print("Analyzer loaded!")
    
    # Initialize state manager with analyzer
    print("Initializing RoadStateManager...")
    app.state.state_manager = get_state_manager(app.state.analyzer)
    print("StateManager ready!")
    
    # Initialize unified CNN-GNN pipeline
    print("Initializing UnifiedRoadAnalyzer (CNN-GNN Pipeline)...")
    app.state.unified_analyzer = UnifiedRoadAnalyzer(
        cnn_model_path=BASE_DIR / "models" / "97.14_modif_resnet18_checkpoint.pth",
        gnn_model_dir=BASE_DIR / "models",
        gnn_data_dir=BASE_DIR / "data"
    )
    print("Unified Pipeline ready!")
    
    # Configure Cloudinary from environment variable
    print("Configuring Cloudinary...")
    cloudinary_url = os.environ.get("CLOUDINARY_URL")
    print(f"  CLOUDINARY_URL found: {bool(cloudinary_url)}")
    if cloudinary_url:
        # Parse URL manually: cloudinary://api_key:api_secret@cloud_name
        from urllib.parse import urlparse
        parsed = urlparse(cloudinary_url)
        cloud_name = parsed.hostname
        api_key = parsed.username
        api_secret = parsed.password
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True
        )
        print(f"Cloudinary configured for cloud: {cloudinary.config().cloud_name}")
    else:
        print("[WARNING] CLOUDINARY_URL not set - uploads will be skipped")
    
    yield
    # Cleanup if needed


# === FastAPI App ===
app = FastAPI(
    title="Road Damage Analysis API",
    description="Pipeline CNN-GNN untuk klasifikasi kerusakan jalan dan analisis dampak",
    version="2.0.0",
    lifespan=lifespan
)

# === CORS Middleware (untuk menerima requests dari n8n.io dan lainnya) ===
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (n8n.io, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# === Schemas ===

class ConditionEnum(str, Enum):
    good = "good"
    fair = "fair"
    poor = "poor"
    very_poor = "very_poor"


class AnalyzeRequest(BaseModel):
    lan: float = Field(..., description="Latitude koordinat jalan", examples=[-7.051533])
    lon: float = Field(..., description="Longitude koordinat jalan", examples=[110.426030])
    condition: ConditionEnum = Field(..., alias="class", description="Kondisi jalan dari Vision Model")


class DamageRequest(BaseModel):
    lan: float = Field(..., description="Latitude koordinat jalan", examples=[-6.9932])
    lon: float = Field(..., description="Longitude koordinat jalan", examples=[110.4203])
    condition: ConditionEnum = Field(..., alias="class", description="Kondisi jalan dari Vision Model")


class FixRequest(BaseModel):
    lan: float = Field(..., description="Latitude jalan yang diperbaiki", examples=[-6.9932])
    lon: float = Field(..., description="Longitude jalan yang diperbaiki", examples=[110.4203])


class AffectedRoadResponse(BaseModel):
    edge: str
    impact: float
    distance: int


class MetadataResponse(BaseModel):
    latitude: float
    longitude: float
    condition: str


class AnalyzeResponse(BaseModel):
    metadata: MetadataResponse
    priority_score: float
    percentile: float
    max_impact: float
    affected_count: int
    affected_roads: List[AffectedRoadResponse]


# ProcessRequest tidak lagi diperlukan karena menggunakan Form dan File upload
# class ProcessRequest(BaseModel):
#     """Request untuk full CNN-GNN pipeline."""
#     lan: float = Field(..., description="Latitude koordinat jalan", example=-6.9932)
#     lon: float = Field(..., description="Longitude koordinat jalan", example=110.4203)
#     img_location: str = Field(..., description="Path ke file gambar jalan", example="d/Test-001.jpg")


class ClassificationResponse(BaseModel):
    condition: str
    confidence: float
    probabilities: dict


class ProcessResponse(BaseModel):
    """Response dari full CNN-GNN pipeline."""
    classification: ClassificationResponse
    metadata: MetadataResponse
    priority_score: float
    percentile: float
    max_impact: float
    affected_count: int
    affected_roads: List[AffectedRoadResponse]
    cloudinary_url: str | None = None


class ProcessUrlRequest(BaseModel):
    """Request dari n8n (FIXED - jangan diubah)."""
    id: str = Field(..., description="UUID report")
    longitude: float = Field(..., description="Longitude koordinat jalan")
    latitude: float = Field(..., description="Latitude koordinat jalan")
    before_img_url: str = Field(..., description="URL gambar dari Cloudinary")


class ProcessUrlResponse(BaseModel):
    """Response dari process-url endpoint."""
    id: str
    classification: ClassificationResponse
    priority_score: float
    percentile: float
    max_impact: float
    affected_count: int



# === Endpoints ===

@app.get("/")
async def root():
    return {
        "message": "Road Damage Analysis API", 
        "version": "2.0.0",
        "endpoints": {
            "POST /process": "Full CNN-GNN pipeline (multipart: lan, lon, imgRaw)",
            "POST /process-url": "Full CNN-GNN pipeline from URL (json: lan, lon, image_url)",
            "POST /analyze": "GNN analysis only (lan, lon, class)",
            "POST /damage": "Report road damage",
            "POST /fix": "Mark road as fixed",
            "POST /status": "Get road status",
            "POST /clear": "Clear all damage records"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_road(request: AnalyzeRequest):
    """
    Analisis kekritisan jalan dan dampak kerusakan.
    
    Input:
    - lan, lon: Koordinat jalan
    - class: Kondisi jalan dari Vision Model (good/fair/poor/very_poor)
    
    Output:
    - Skor kekritisan jalan
    - Dampak ke jalan sekitar
    - Prioritas perbaikan
    - Rekomendasi
    """
    try:
        analyzer: RoadDamageAnalyzer = app.state.analyzer
        result = analyzer.analyze_from_vision(
            latitude=request.lan,
            longitude=request.lon,
            condition=request.condition.value
        )
        
        return result.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process", response_model=ProcessResponse)
async def process_road_image(
    lan: float = Form(..., description="Latitude koordinat jalan", examples=[-6.9932]),
    lon: float = Form(..., description="Longitude koordinat jalan", examples=[110.4203]),
    imgRaw: UploadFile = File(..., description="File gambar jalan (JPG/PNG)")
):
    """
    Full Pipeline: Image → CNN Classification → GNN Impact Analysis
    
    Input (Multipart Form Data):
    - lan: Latitude koordinat jalan
    - lon: Longitude koordinat jalan  
    - imgRaw: File gambar jalan (upload)
    
    Output:
    - classification: Hasil klasifikasi CNN (condition, confidence)
    - priority_score: Skor prioritas perbaikan
    - affected_roads: Jalan-jalan yang terdampak
    """
    temp_file = None
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if imgRaw.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {imgRaw.content_type}. Allowed: {allowed_types}"
            )
        
        # Save uploaded file to temp location
        suffix = Path(imgRaw.filename).suffix if imgRaw.filename else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        shutil.copyfileobj(imgRaw.file, temp_file)
        temp_file.close()
        
        # Process with unified analyzer
        unified_analyzer: UnifiedRoadAnalyzer = app.state.unified_analyzer
        result = unified_analyzer.analyze(
            lan=lan,
            lon=lon,
            img_location=temp_file.name
        )
        
        # Save image to outputnye directory with format {time}{lon}{lat}.jpg
        output_dir = BASE_DIR / "outputnye"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Format lon and lat untuk nama file (ganti titik dan minus)
        lon_str = str(lon).replace(".", "_").replace("-", "m")
        lat_str = str(lan).replace(".", "_").replace("-", "m")
        output_filename = f"{timestamp}{lon_str}{lat_str}.jpg"
        output_path = output_dir / output_filename
        
        # Copy temp file ke output directory
        shutil.copy2(temp_file.name, output_path)
        
        # Upload to Cloudinary (if configured)
        cloudinary_url = None
        if os.environ.get("CLOUDINARY_URL"):
            try:
                cloud_result = cloudinary.uploader.upload(
                    temp_file.name,
                    public_id=output_filename.replace(".jpg", ""),
                    folder="road_damage",
                    tags=["road_damage", result["classification"]["condition"]],
                    resource_type="image"
                )
                cloudinary_url = cloud_result.get("secure_url")
                print(f"[Cloudinary] Uploaded: {cloudinary_url}")
            except Exception as cloud_err:
                print(f"[Cloudinary] Upload failed: {cloud_err}")
        
        # Add cloudinary_url to result
        result["cloudinary_url"] = cloudinary_url
        
        return result
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink()



@app.post("/process-url", response_model=ProcessUrlResponse)
async def process_from_url(request: ProcessUrlRequest):
    """
    Process image dari n8n webhook automation.
    
    Flow:
    1. Download image dari before_img_url ke folder temp/
    2. Process dengan CNN-GNN pipeline
    3. Return hasil klasifikasi
    4. Hapus file temp setelah selesai
    
    Input (dari n8n - FIXED):
    - id: UUID report
    - longitude, latitude: Koordinat jalan
    - before_img_url: URL gambar dari Cloudinary
    
    Output:
    - classification: Hasil klasifikasi CNN
    - priority_score, percentile: Skor GNN
    """
    
    # Create temp directory if not exists
    temp_dir = BASE_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Generate unique temp filename
    temp_filename = f"{request.id}.jpg"
    temp_path = temp_dir / temp_filename
    
    try:
        # 1. Download image from Cloudinary URL
        print(f"[process-url] Downloading image from: {request.before_img_url}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(request.before_img_url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image: HTTP {response.status_code}"
                )
            image_data = response.content
        
        # Save to temp folder
        with open(temp_path, "wb") as f:
            f.write(image_data)
        print(f"[process-url] Image saved to: {temp_path}")
        
        # 2. Process with unified analyzer (CNN-GNN)
        unified_analyzer: UnifiedRoadAnalyzer = app.state.unified_analyzer
        result = unified_analyzer.analyze(
            lan=request.latitude,
            lon=request.longitude,
            img_location=str(temp_path)
        )
        print(f"[process-url] Classification: {result['classification']['condition']}")
        
        return ProcessUrlResponse(
            id=request.id,
            classification=ClassificationResponse(
                condition=result["classification"]["condition"],
                confidence=result["classification"]["confidence"],
                probabilities=result["classification"]["probabilities"]
            ),
            priority_score=result["priority_score"],
            percentile=result["percentile"],
            max_impact=result["max_impact"],
            affected_count=result["affected_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[process-url] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 4. Cleanup: Delete temp file after processing
        if temp_path.exists():
            temp_path.unlink()
            print(f"[process-url] Temp file deleted: {temp_path}")



# === State Management Endpoints ===

@app.post("/damage")
async def report_damage(request: DamageRequest):
    """
    Report jalan rusak (simpan ke state).
    
    Menyimpan kerusakan dan menghitung cascade effects ke jalan sekitar.
    """
    try:
        manager: RoadStateManager = app.state.state_manager
        record = manager.report_damage(
            lat=request.lan,
            lon=request.lon,
            condition=request.condition.value
        )
        
        return {
            "status": "reported",
            "edge": f"{record.u}→{record.v}",
            "edge_idx": record.edge_idx,
            "condition": record.condition,
            "severity": record.severity
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fix")
async def fix_road(request: FixRequest):
    """
    Jalan sudah diperbaiki.
    
    Menghapus dari state dan menghitung ulang cascade effects.
    """
    try:
        manager: RoadStateManager = app.state.state_manager
        success = manager.mark_fixed(request.lan, request.lon)
        
        if success:
            return {"status": "fixed"}
        else:
            return {"status": "not_found", "message": "No damaged road found at this location"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/damaged")
async def get_damaged():
    """
    List semua jalan rusak beserta summary.
    """
    try:
        manager: RoadStateManager = app.state.state_manager
        return manager.get_summary()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/status")
async def get_status(
    lan: float = Query(..., description="Latitude koordinat jalan"),
    lon: float = Query(..., description="Longitude koordinat jalan")
):
    """
    Status jalan tertentu.
    
    Menampilkan skor original, skor saat ini, dan sumber impact dari jalan rusak lain.
    """
    try:
        manager: RoadStateManager = app.state.state_manager
        status = manager.get_road_status(lan, lon)
        
        return status.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_all():
    """
    Clear semua damage records dan reset ke baseline.
    """
    try:
        manager: RoadStateManager = app.state.state_manager
        manager.clear_all()
        
        return {"status": "cleared", "message": "All damage records cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

