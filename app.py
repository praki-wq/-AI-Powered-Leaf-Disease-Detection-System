from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import logging
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PIL import Image
from database import engine
from models import Base
from fastapi import Form
from database import SessionLocal
from models import User
from fastapi.responses import RedirectResponse
from models import DetectionHistory
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
import datetime
import random
import tensorflow as tf
import numpy as np
from fastapi import Request
from fastapi.responses import HTMLResponse
from tensorflow.keras.applications.efficientnet import preprocess_input
class_names = [

    "fungus",

    "healthy leaf",

    "leaf spot",

    "mosaic virus",

    "rust leaf"

]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Leaf Disease Detection API", version="1.1.0")
Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- IMAGE VALIDATION ---------------- #


def validate_image(image):
    """
    Validate uploaded image quality and dimensions
    """

    width, height = image.size

    # Minimum resolution check
    if width < 100 or height < 100:
        return False

    return True


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):

    return templates.TemplateResponse(request=request, name="login.html")


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):

    return templates.TemplateResponse(request=request, name="signup.html")


@app.post("/signup")
async def signup_user(
    username: str = Form(...), email: str = Form(...), password: str = Form(...)
):

    db = SessionLocal()

    # Check existing email
    existing_user = db.query(User).filter(User.email == email).first()

    if existing_user:

        return {"message": "Email already exists"}

    # Create new user
    new_user = User(username=username, email=email, password=password)

    db.add(new_user)

    db.commit()

    db.close()

    return RedirectResponse(
    url="/",
    status_code=303
)



@app.api_route("/login", methods=["GET", "POST"])
async def login_user(
    request: Request,
    email: str = Form(None),
    password: str = Form(None)
):

    # ADD THIS PART
    if request.method == "GET":

        return templates.TemplateResponse(
            request=request,
            name="login.html"
        )

    db = SessionLocal()

    user = db.query(User).filter(
        User.email == email,
        User.password == password
    ).first()

    db.close()

    if user:

        return templates.TemplateResponse(

            request=request,

            name="dashboard.html",

            context={

                "username": user.username

            }
        )

    return {
        "message": "Invalid email or password"
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):

    return templates.TemplateResponse(

        request=request,

        name="dashboard.html",

        context={

            "username": "Prajwal"

        }
    )

@app.get("/detect", response_class=HTMLResponse)
async def detect_page(request: Request):

    return templates.TemplateResponse(

        request=request,

        name="detect.html",

        context={

            "username": "Prajwal"

        }
    )

    
@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):

    db = SessionLocal()

    records = db.query(
        DetectionHistory
    ).all()

    db.close()

    return templates.TemplateResponse(

        request=request,

        name="history.html",

        context={

            "records": records,

            "username": "Prajwal"

        }
    )

@app.get("/ai-analysis", response_class=HTMLResponse)
async def ai_analysis(request: Request):

    return templates.TemplateResponse(

        request=request,

        name="analysis.html",

        context={

            "username": "Prajwal"

        }
    )

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):

    return templates.TemplateResponse(

        request=request,

        name="about.html",

        context={

            "username": "Prajwal"

        }
    )
# ---------------- API ENDPOINT ---------------- #


@app.post("/disease-detection-file")
async def disease_detection_file(request: Request, file: UploadFile = File(...)):
    """
    Detect diseases in leaf images using uploaded image file
    """
    from utils import convert_image_to_base64_and_test

    try:

        logger.info("Received image file for disease detection")

        # Read uploaded file
        contents = await file.read()
        import base64

        image_base64 = base64.b64encode(contents).decode("utf-8")

        # Open image using PIL
        try:
            image = Image.open(BytesIO(contents))
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "disease_detected": False,
                    "disease_type": "invalid_image",
                    "symptoms": ["Uploaded file is not a valid image"],
                    "treatment": ["Please upload a valid leaf image"],
                },
            )

        # Validate image quality
        if not validate_image(image):

            return JSONResponse(
                status_code=400,
                content={
                    "disease_detected": False,
                    "disease_type": "invalid_image",
                    "symptoms": ["Image resolution is too low"],
                    "treatment": ["Upload a clearer and higher quality leaf image"],
                },
            )

        # Process image using AI model
        # Process image using AI model
        result = convert_image_to_base64_and_test(contents)
        print(result)
        model = tf.keras.models.load_model(
            "model/leaf_disease_model.h5"
        )
# -------- REAL CONFIDENCE -------- #
        image = image.convert("RGB")
        
        image = image.resize((224,224))

        img_array = np.array(image)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )
        
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)

        confidence = np.max(prediction) * 100

        predicted_class = np.argmax(prediction)
        
        predicted_disease = class_names[predicted_class]

        print(result)

        if result is None:

            result = {

        "symptoms": [
            "AI analysis unavailable"
            ],

        "treatment": [
            "Please try another clearer leaf image"
            ]
            }

        db = SessionLocal()

        history = DetectionHistory(

            disease=predicted_disease,
            image=image_base64,

             created_at=str(
                 datetime.datetime.now()
            )
        )

        db.add(history)

        db.commit()

        db.close()

        logger.info("Disease detection completed successfully")

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={

                "image": image_base64,

                "disease": predicted_disease,
                "confidence":
                f"{confidence:.2f}%",

                "symptoms": result.get(
                    "symptoms",
                    []
            ),

                "treatment": result.get(
                    "treatment",
                    []
            ),
        }
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.error(f"Error in disease detection: {str(e)}")

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
