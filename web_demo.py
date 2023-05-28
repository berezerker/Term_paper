import os
import sys
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import onnxruntime as ort
import torchvision.transforms as tt
import time
import numpy as np

sys.path.append("../")

root_dir = os.path.dirname(__file__)

EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

@st.cache(allow_output_mutation=True)
def create_inference():
    ort_session = ort.InferenceSession("saved_models/exported/final_25_05_23_export_script.onnx", providers = EP_list)
    dummy_tensor = np.random.rand(1,3,512,512).astype("float32")
    outputs = ort_session.run(
            None,
            {"actual_input": dummy_tensor},
        )
    return ort_session

def predict(img, session):
    start_time = time.time()
    outputs = ort_session.run(
        None,
        {"actual_input": img},
    )
    elapsed_time = 1000 * (time.time() - start_time)
    return elapsed_time, outputs[0][0]

ort_session = create_inference()

st.header('Web demo for detection of degraded images')
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is None:
    st.text('Waiting for upload...')
else:
    slot = st.empty()

    img = Image.open(uploaded_file)
    st.image(img, "your image")
    to_tensor = tt.Compose([tt.Resize((512, 512)), tt.ToTensor()])
    img = to_tensor(img).unsqueeze(0).numpy()
    elapsed_time, outputs = predict(img, ort_session)
    st.text(f"Inference finished. Time elapsed: {elapsed_time:.1f}ms.")
    st.text(f'Image has blur with probability {outputs[0] * 100:.1f}%')
    st.text(f'Image has moire with probability {outputs[1] * 100:.1f}%')
    st.text(f'Image has haze with probability {outputs[2] * 100:.1f}%')
    st.text(f'Image has rain with probability {outputs[3] * 100:.1f}%')
