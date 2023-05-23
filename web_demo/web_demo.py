import os
import sys
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import onnxruntime as ort
import torchvision.transforms as tt

sys.path.append("../")

root_dir = os.path.dirname(__file__)


def main():
    st.header("Detection of blurred image demo")
    st.write("Upload your image for detecting:")

    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, "your image")
        to_tensor = tt.Compose([tt.Resize((512, 512)), tt.ToTensor()])
        img = to_tensor(img).unsqueeze(0).numpy()
        ort_session = ort.InferenceSession("../saved_models/exported/final_19_05_23.onnx")

        outputs = ort_session.run(
            None,
            {"actual_input": img},
        )
        outputs = outputs[0][0]
        st.text(f"Image has blur with probability {outputs[0] * 100:.1f}%")
        st.text(f'Image has moire with probability {outputs[1] * 100:.1f}%')
        st.text(f'Image has haze with probability {outputs[2] * 100:.1f}%')
        st.text(f'Image has rain with probability {outputs[3] * 100:.1f}%')


if __name__ == "__main__":
    main()
