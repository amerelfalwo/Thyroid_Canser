import os
import warnings
import uuid
import shutil

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.metadata.schema import MetadataInput
from app.disease.schema import ThyroidInput
from app.metadata.model import predict_metadata
from app.disease.model import predict_thyroid
from app.segmentation.model import segment_image

from pyngrok import ngrok
import uvicorn


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
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return str(obj)


@app.post("/predict/disease")
def predict_thyroid_disease(data: ThyroidInput):
    result = predict_thyroid(data.dict())
    return safe_return(result)


@app.post("/predict/segmentation-metadata")
async def segmentation_metadata_predict(
    file: UploadFile = File(...),
    age: int = None,
    gender: int = None,
    FNAC: int = None
):

    img_name = f"{uuid.uuid4()}.png"
    img_path = os.path.join(UPLOAD_DIR, img_name)

    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    seg_result = segment_image(img_path)
    seg_result = safe_return(seg_result)

    tirads_value = seg_result.get("tirads")

    if tirads_value is None:
        return {"error": "Failed to calculate TIRADS"}

    metadata_input = {
        "age": age,
        "gender": gender,
        "TIRADS": tirads_value,
        "FNAC": FNAC
    }

    metadata_result = predict_metadata(metadata_input)
    metadata_result = safe_return(metadata_result)

    return {
        "segmentation": seg_result,
        "metadata_prediction": metadata_result
    }



if __name__ == "__main__":

    ngrok.kill()

    ngrok.set_auth_token("39hlJNDgDaQbCi7nqxFzOs1HqEv_64YxA7yfnuwPRd8svnR4E")

    public_url = ngrok.connect(addr=8500)

    print("\nPublic URL:")
    print(public_url)
    print("\nSwagger:")
    print(str(public_url) + "/docs")

    uvicorn.run(app, host="0.0.0.0", port=8500)
