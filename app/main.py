import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid, shutil

from app.metadata.schema import MetadataInput
from app.disease.schema import ThyroidInput
from app.metadata.model import predict_metadata
from app.disease.model import predict_thyroid
from app.segmentation.model import segment_image

app = FastAPI(title="Medical AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def safe_return(obj):
    if isinstance(obj, dict):
        return {k: float(v) if hasattr(v, "item") else v for k, v in obj.items()}
    elif hasattr(obj, "tolist"):  
        return obj.tolist()
    elif hasattr(obj, "item"):  #
        return obj.item()
    else:
        return str(obj)

@app.post("/predict/metadata")
def metadata_predict(data: MetadataInput):
    result = predict_metadata(data.dict())
    return safe_return(result)

@app.post("/predict/disease")
def predict_thyroid_disease(data: ThyroidInput):
    result = predict_thyroid(data.dict())
    return safe_return(result)

@app.post("/predict/segmentation")
async def segmentation_predict(file: UploadFile = File(...)):
    img_name = f"{uuid.uuid4()}.png"
    img_path = os.path.join(UPLOAD_DIR, img_name)
    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = segment_image(img_path)
    return safe_return(result)
