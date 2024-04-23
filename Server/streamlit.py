import streamlit as st
import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d


def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __class_name_to_number
    global __class_number_to_name
    with open(r"C:\Users\sumit\Desktop\Code\Project\Image classification\Server\artifacts\class_dictonary.json","r") as f:
        __class_name_to_number=json.load(f)
        __class_number_to_name={v:k for k,v in __class_name_to_number.items()}
    global __model
    if __model is None:
        with open(r'C:\Users\sumit\Desktop\Code\Project\Image classification\Server\artifacts\saved_model.pkl','rb') as f:
            __model=joblib.load(f)
        print("Loading Saved Artifacts done...")
# Load saved artifacts


__class_name_to_number={}
__class_number_to_name={}
__model=None

def  class_number_to_name(class_num):
    return  __class_number_to_name[class_num]

# Define image classification function
def classify_image(file_path):
    imgs=get_cropped_image_if_2_eyes(file_path)
    result=[]
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        len_image_array=32*32*3+32*32
        final=combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class':class_number_to_name( __model.predict(final)[0]),
            'class_probability':np.round(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary':__class_name_to_number
        })
            
    return result

# Define function to convert base64 image to numpy array
def get_cv2_image_from_base64_string(b64str):
    # Code for converting base64 image to numpy array
    try:
        encoded_data = b64str.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        else:
            print("Error: Image not loaded correctly.")
            return None
    except Exception as e:
        print("Error decoding base64 string:", e)
        return None

face_cascade=cv2.CascadeClassifier(r"C:\Users\sumit\Desktop\Code\Project\Image classification\Model\opencv\haarcascades\haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier(r"C:\Users\sumit\Desktop\Code\Project\Image classification\Model\opencv\haarcascades\haarcascade_eye.xml")

# Define function to detect faces and eyes and return cropped faces
def get_cropped_image_if_2_eyes(image_path):
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    

# Define function to upload image file
def upload_image_file():
    image_file = st.file_uploader("Upload an image",type=['jpeg','png','webp'])
    st.download_button(label="Upload", data=data, file_name=uploaded_file.name)
    if image_file is not None:
        image_data = image_file.getvalue()
        img=cv2.imread(image_data)
        print(img)
        image_base64_data = base64.b64encode(image_data).decode()
        st.write("Classifying image...")
        result = classify_image(image_data)
        st.write(result)

# Define main function
def main():
    st.title("Cricket Player Image Prediction")
    st.write("Upload a photo of a cricket player to predict the player's name.")
    load_saved_artifacts()
    upload_image_file()
    

if __name__ == "__main__":
    main()