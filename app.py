from flask import Flask,request,jsonify,render_template
import os
import torch
from torchvision import models,transforms
import numpy as np
from PIL import Image
import io
import face_recognition
import cv2
import face_recognition
from torch import nn



def predict_transfer(pil_image,model):
    # load the image and return the predicted breed
    mean_train_set,std_train_set = [0.487,0.467,0.397],[0.235,0.23,0.23]
    
    image_transforms= transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean_train_set,std_train_set)])
    image_tensor = image_transforms(pil_image);
    image_tensor.unsqueeze_(0)
#     if use_cuda:
#         image_tensor = image_tensor.cuda()
    model.eval()
    output = model(image_tensor)
    _,class_idx=torch.max(output,dim=1)
    class_name={0: 'Apple___Apple_scab',
 1: 'Apple___Black_rot',
 2: 'Apple___Cedar_apple_rust',
 3: 'Apple___healthy',
 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew',
 6: 'Cherry_(including_sour)___healthy',
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___healthy',
 11: 'Grape___Black_rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___healthy',
 15: 'Orange___Haunglongbing_(Citrus_greening)',
 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy',
 18: 'Pepper,_bell___Bacterial_spot',
 19: 'Pepper,_bell___healthy',
 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight',
 22: 'Potato___healthy',
 23: 'Raspberry___healthy',
 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew',
 26: 'Strawberry___Leaf_scorch',
 27: 'Strawberry___healthy',
 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight',
 30: 'Tomato___Late_blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Tomato___healthy'}
    return class_name[class_idx]


app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
	return render_template("index.html")


@app.route('/about')
def render_about_page():
	return render_template('about.html')

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
	if request.method == "POST":
		if request.files:
			image = request.files["image"]
			image_bytes = image.read()
			image_extensions=['ras', 'xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm', 'xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm']
# 			return jsonify('This a disease picture of ')

#             		image = request.files["image"]
			
 			
			
# 	if request.method == "POST":
# 		file = request.files['image']
    			
#     			if image.filename.split('.')[1] not in image_extensions:
# 				return jsonify('Please upload an appropriate image file')
# 			model = torch.load("Model/model_plant.pt")
	
# 			image_bytes = image.read()
#     			pil_image = Image.open(io.BytesIO(image_bytes))
    
#     			nparr = np.frombuffer(image_bytes, np.uint8)
#     			img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    
#         		disease = predict_transfer(pil_image,model)
       	
    
    
	
	



if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))

