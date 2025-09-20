import streamlit as st
import torch
from PIL import Image

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

model = load_model()

st.title("🚗🏍 Car or Bike Detection App")
st.write("Upload an image and the app will detect cars or bikes using YOLOv5.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(img)
    results.render()  # draws boxes & labels
    detected_img = Image.fromarray(results.ims[0])

    st.image(detected_img, caption="Detected Objects", use_column_width=True)

    # Extract labels
    labels = results.pandas().xyxy[0]['name'].tolist()
    car_bike_labels = [label for label in labels if label in ["car", "motorbike", "bicycle"]]

    if car_bike_labels:
        st.success(f"✅ Detected: {', '.join(car_bike_labels)}")
    else:
        st.warning("No cars or bikes detected.")
