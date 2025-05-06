import os
from flask import Flask, request, jsonify,Response
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import torch    
import torchvision.transforms as transforms
from model import CNN_NeuralNet
from ultralytics import YOLO
import cv2
import base64
from io import BytesIO
from bson.json_util import dumps
import google.generativeai as genai
from inference_sdk import InferenceHTTPClient
from flask import Flask, request, jsonify
import requests
import numpy as np
from collections import defaultdict
from geopy.geocoders import Nominatim
from pymongo import MongoClient
import datetime
from bson import Binary

import os
API_KEY = os.getenv("API_KEY")

client = MongoClient("mongodb://localhost:27017/")
db = client["kuickhack2025"]
collection = db["historical_data"]


DIFY_API_URL = "https://api.dify.ai/v1/workflows/run"
DIFY_API_KEY = os.getenv("API_KEY_DIFY") 

geolocator = Nominatim(user_agent="my-unique-app")

def serialize_document(doc):
    return {
        '_id': str(doc.get('_id')),
        'name': doc.get('name'),
        'date': doc.get('date').isoformat() if doc.get('date') else '',
        'percentOfIllness': doc.get('percentOfIllness'),
        'geminiInfo': doc.get('gemini_info', {}),
        'image': f"data:image/png;base64,{base64.b64encode(doc['image']).decode()}" if 'image' in doc else None
    }

translations = {
    "plants": {
        "Apple": "Яблоня",
        "Blueberry": "Голубика",
        "Cherry_(including_sour)": "Вишня (включая кислую)",
        "Corn_(maize)": "Кукуруза",
        "Grape": "Виноград",
        "Orange": "Апельсин",
        "Peach": "Персик",
        "Pepper,_bell": "Перец (сладкий)",
        "Potato": "Картофель",
        "Raspberry": "Малина",
        "Rice": "Рис",
        "Soybean": "Соя",
        "Squash": "Кабачок",
        "Strawberry": "Клубника",
        "Tomato": "Томат",
        "Wheat": "Пшеница"
    },
    "conditions": {
        "Apple_scab": "Парша яблони",
        "Black_rot": "Чёрная гниль",
        "Cedar_apple_rust": "Ржавчина яблони",
        "healthy": "Здоровое растение",
        "Powdery_mildew": "Мучнистая роса",
        "Cercospora_leaf_spot Gray_leaf_spot": "Пятнистость листьев",
        "Common_rust_": "Обычная ржавчина",
        "Northern_Leaf_Blight": "Северный ожог листьев",
        "Esca_(Black_Measles)": "Эска (чёрная корь)",
        "Leaf_blight_(Isariopsis_Leaf_Spot)": "Ожог листьев (пятнистость)",
        "Haunglongbing_(Citrus_greening)": "Зелёная болезнь цитрусовых",
        "Bacterial_spot": "Бактериальная пятнистость",
        "Early_blight": "Ранняя фитофторозная пятнистость",
        "Late_blight": "Поздний фитофтороз",
        "BrownSpot": "Бурая пятнистость",
        "Hispa": "Гиспа (вредитель)",
        "LeafBlast": "Взрыв листьев",
        "Leaf_scorch": "Подгорание листьев",
        "Leaf_Mold": "Листовая плесень",
        "Septoria_leaf_spot": "Септориозная пятнистость листьев",
        "Spider_mites Two-spotted_spider_mite": "Паутинные клещи (двухпятнистый)",
        "Target_Spot": "Таргетная пятнистость",
        "Tomato_Yellow_Leaf_Curl_Virus": "Жёлтая скрученность листьев томата (вирус)",
        "Tomato_mosaic_virus": "Мозаичный вирус томата",
        "septoria": "Септориоз",
        "stripe_rust": "Полосатая ржавчина"
    }
}


def gpt(inputs=None, user_id="abc-123", response_mode="blocking"):
    url = DIFY_API_URL
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": {
            "input_user": inputs
        },
        "response_mode": response_mode,
        "user": user_id
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["data"].get("outputs").get("text")


  

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Labels
labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Rice___BrownSpot', 'Rice___Healthy', 'Rice___Hispa', 'Rice___LeafBlast',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Wheat__Healthy',
    'Wheat__septoria', 'Wheat__stripe_rust'
]

# Load models
model_class = CNN_NeuralNet(3, 45)
model_class.load_state_dict(torch.load('my_with_wheat.pth', map_location=torch.device('cpu')))
model_class.eval()

model_detection = YOLO("last_model.pt")
model_whole_leaf = YOLO("best_whole_leaf_detection.pt")

# Transforms
transform_class = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

transform_detect = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Utility: convert OpenCV image to base64
def convert_image_to_base64(image_np):
    image_np=np.array(image_np)
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def draw_classes(result,img):
        img=img.convert("RGB")   
        draw = ImageDraw.Draw(img)
        area_by_class = defaultdict(float)
        showed_labels=set()
        print(result[0]["predictions"]["predictions"])
        for det in result[0]["predictions"]["predictions"]:
            x0 = det["x"] - det["width"] / 2
            y0 = det["y"] - det["height"] / 2
            x1 = det["x"] + det["width"] / 2
            y1 = det["y"] + det["height"] / 2
            if det['class'] not in showed_labels:
                label = f"{det['class']}"
                draw.text((x0+5, y0 + 10), label, fill=(200,0,0))
                showed_labels.add(label)
                
            

            # Draw rectangle and label
            draw.rectangle([x0, y0, x1, y1], outline=(0,0,200), width=2)
            area = det["width"] * det["height"]
            area_by_class[det["class"].lower()] += area
        return img,area_by_class



# Classification endpoint
@app.route('/predict-class', methods=['POST'])
def predict_class():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    img = Image.open(file.stream).convert('RGB')
    img = transform_class(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model_class(img)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    return jsonify({'prediction': labels[predicted_class]})

# Detection + Gemini endpoint
@app.route('/predict-detect', methods=['POST'])
def predict_detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    file = request.files['image']
    predicted_label = request.form["diagnosis"]

    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')

        # Roboflow Inference
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=os.getenv("API_KEY_ROBOFLOW")
        )

        result = client.run_workflow(
            workspace_name="fusnoper",
            workflow_id="detect-count-and-visualize-2",
            images={ "image": img }
        )

        img,areas=draw_classes(result,img)
        encoded_image = convert_image_to_base64(img)
        fruit, result= predicted_label.split("___")
        up=0
        bottom=0
        for key in areas.keys():
            if key.lower()==fruit.lower():
                bottom += areas[key]
            else:
                up+= areas[key]
        print(areas)
        if up==0 or bottom==0:
            area="Невозможно определить соотношение"
        else:
            area=str(round(up/bottom*100,2))+"%"


        # Gemini Info
        try:
            location = geolocator.geocode(request.form["latitude"]+","+request.form["longitude"])
            print(location)
            fruit,result= translations["plants"][fruit], translations["conditions"][result]
            prompt = f"Мой {fruit} лист страдает от {result}. Я сейчас нахожусь {location}"
            response = gpt(prompt)
            causes_and_solutions = response.strip()
        except Exception as e:
            print(f"Error with Gemini API: {str(e)}")
            causes_and_solutions = "Не удалось получить дополнительную информацию о заболевании."

        return jsonify({
            'prediction': encoded_image,
            'causes': causes_and_solutions,
            'solutions': causes_and_solutions,
            'percent_of_illnesses': area
        })

    except Exception as e:
        print(f"Error in predict_detect: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500


# Whole-leaf detection (you can optionally add Gemini here too)
@app.route('/predict-detect-whole-leaf', methods=['POST'])
def predict_detect_full():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    file = request.files['image']
    predicted_label = request.form["diagnosis"]

    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    img = Image.open(file.stream).convert('RGB')
    client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=os.getenv("API_KEY_ROBOFLOW")
        )   

    result = client.run_workflow(
            workspace_name="fusnoper",
            workflow_id="detect-count-and-visualize-2",
            images={ "image": img }
        )
    

    img,areas=draw_classes(result,img)
    fruit, result= predicted_label.split("___")
    up=0
    bottom=0
    for key in areas.keys():
        if key.lower()==fruit.lower():
            bottom += areas[key]
        else:
            up+= areas[key]
    print(areas)
    if up==0 or bottom==0:
        area="Невозможно определить соотношение"
    else:
        area=str(round(up/bottom*100,2))+"%"

    encoded_image = convert_image_to_base64(img)

    return jsonify({'prediction': encoded_image,'percent_of_illnesses': area})


@app.route('/save-in-depth-analysis', methods=['POST'])
def save_analysis():
    try:
        data = request.get_json()
        
        # Decode the base64 image (format: "data:image/png;base64,....")
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        print(data)
        record = {
            "name": data["name"],
            "percentOfIllness": data["percentOfIlness"],
            "gemini_info": data["geminiInfo"],
            "image": Binary(image_bytes),  # storing image in binary
            "date": datetime.datetime.fromisoformat(data["date"].replace('Z', '')),
        }

        collection.insert_one(record)
        return jsonify({"message": "Saved successfully"}), 200

    except Exception as e:
        print("Error saving to DB:", e)
        return jsonify({"error": "Failed to save"}), 500
    

@app.route('/retrieve_all_documents')
def get_documents():
    docs = list(collection.find())
    return jsonify([serialize_document(doc) for doc in docs])

    
# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
