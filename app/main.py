from fastapi import FastAPI, UploadFile, File
import uuid, os, shutil
from app.metadata.schema import MetadataInput
from app.metadata.model import predict_metadata
from app.segmentation.model import segment_image

app = FastAPI(title="Medical AI API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict/metadata")
def metadata_predict(data: MetadataInput):
    return predict_metadata(data.dict())

@app.post("/predict/segmentation")
async def segmentation_predict(file: UploadFile = File(...)):
    img_name = f"{uuid.uuid4()}.png"
    img_path = os.path.join(UPLOAD_DIR, img_name)

    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = segment_image(img_path)

    return {
        "mask_path": result["mask_path"],
        "overlay_path": result["overlay_path"],
        "roi_path": result["roi_path"],
        "bbox": result["bbox"]
    }
