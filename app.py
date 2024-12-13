import streamlit as st
import cv2
from sklearn.cluster import KMeans
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import requests

# 색상 추출 함수
def extract_colors(image, num_colors=5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors).fit(image)
    return kmeans.cluster_centers_

# ResNet50 모델 불러오기
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

# Stable Diffusion API 호출 함수
def generate_character(prompt):
    api_url = "https://api.stability.ai/v1/generation"
    headers = {"Authorization": "sk-f03E0V3fw8MS4tV6OA5Bkmb7FgYxpLpHTxaAKkp3fBnj9PWM"}
    data = {
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "samples": 1
    }
    response = requests.post(api_url, headers=headers, json=data)
    return response.content

# Streamlit UI 구성
st.title("나만의 브랜드 캐릭터 생성")
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png"])

if uploaded_file:
    # 이미지 처리 및 분석
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_data = cv2.imdecode(file_bytes, 1)
    st.image(image_data, caption="업로드된 이미지")

    # 색상 추출
    colors = extract_colors(image_data)
    st.write("추출된 주요 색상:")
    for color in colors:
        st.write(f"RGB 값: {color}")

    # 특징 추출
    file_path = "uploaded_image.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    features = extract_features(file_path)
    st.write("추출된 이미지 특징 벡터:", features)

    # 캐릭터 생성
    prompt = "A cute pink cat with a bright bow in a soft background"
    generated_image = generate_character(prompt)
    with open("generated_character.jpg", "wb") as f:
        f.write(generated_image)
    st.image("generated_character.jpg", caption="생성된 캐릭터")
