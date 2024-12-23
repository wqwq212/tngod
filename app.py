import streamlit as st
import cv2
from sklearn.cluster import KMeans
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import requests
from PIL import Image
import io
import os

# 색상 추출 함수
def extract_colors(image, num_colors=5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors).fit(image)
    return kmeans.cluster_centers_

# ResNet50 모델 불러오기
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

import requests

# API 정보
api_url = "https://api.stability.ai/v1beta/generation/stable-diffusion-v1-5/text-to-image"
headers = {"Authorization": "sk-o7AiiuKV7gLU0zCjhne3fKtUTmyRL9gVgKpc2VgrnGp3CnOa"}  # 실제 API 키로 변경
data = {
    "prompt": "A cute pink cat with a bright bow in a soft background",
    "width": 512,
    "height": 512,
    "samples": 1,
    "cfg_scale": 7.5,
    "sampler": "ddim"
}

# POST 요청
response = requests.post(api_url, headers=headers, json=data)

# 응답 처리
if response.status_code == 200:
    with open("generated_character.jpg", "wb") as f:
        f.write(response.content)
    print("이미지 생성 성공!")
else:
    print(f"API 호출 실패! 상태 코드: {response.status_code}")
    print("응답 내용:", response.text)


# 이미지 데이터 유효성 확인
def validate_image(image_data):
    try:
        Image.open(io.BytesIO(image_data)).verify()
        return True
    except Exception as e:
        st.error("이미지 데이터가 유효하지 않습니다.")
        st.write("오류 내용:", e)
        return False

# Streamlit UI 구성
st.title("나만의 캐릭터 생성")
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
        st.write(f"RGB 값: {color.astype(int)}")

    # 특징 추출
    file_path = "uploaded_image.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    features = extract_features(file_path)
    st.write("추출된 이미지 특징 벡터:", features)

   # 캐릭터 생성 함수 정의
def generate_character(prompt):
    api_url = "https://api.stability.ai/v1beta/generation/stable-diffusion-v1-5/text-to-image"
    headers = {"Authorization": "sk-o7AiiuKV7gLU0zCjhne3fKtUTmyRL9gVgKpc2VgrnGp3CnOa"}  # 실제 API 키로 변경
    data = {
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "samples": 1,
        "cfg_scale": 7.5,
        "sampler": "ddim"
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f"API 호출 실패: {response.status_code}, {response.text}")

    return response.content

# Streamlit UI
import streamlit as st

st.title("나만의 브랜드 캐릭터 생성")
prompt = st.text_input("캐릭터 생성 프롬프트를 입력하세요", "A cute pink cat with a bright bow in a soft background")
if st.button("캐릭터 생성"):
    try:
        generated_image = generate_character(prompt)
        with open("generated_character.jpg", "wb") as f:
            f.write(generated_image)
        st.image("generated_character.jpg", caption="생성된 캐릭터")
    except Exception as e:
        st.error(f"오류 발생: {e}")
