# Term Paper
Final project for HSE program

In this repository you can find jupyter notebooks that were used while developping the solution, as well as the web application for showing the model inference.
This project aimed to develop a multi-label classification model for 4 types of degradations: blur, moire, haze and rain.

# Architecture

We based the architecture of our model on [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

# Metrics

We collected metrics on each class separately and a total F1-score. You can see obtained metrics here.
| class/metric     | precision      | recall     | F1-score |
| -------------    | -------------  | --------   |          |

# Model
The final model is located in folder saved_models/.

# Requirements
install the requirements by running 
```code
pip install -r requirements.txt
```

# Web Demo
In Order to run the web demonstration, type the following
```code
streamlit run web_demo.py
```
