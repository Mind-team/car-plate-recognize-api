from fastapi import FastAPI, UploadFile, File
from car_plate_recognition_model import get_plate
import cv2



app = FastAPI()


@app.post("/api/camera/get-carplate")
def register_entry(img: UploadFile = File(...)):
    carplate_img = cv2.imread(img.filename)
    return {"plate": get_plate(carplate_img)}  # ResponseTime: 2.32s

