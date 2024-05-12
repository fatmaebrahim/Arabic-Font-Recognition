import gradio as gr
import joblib
import cv2
import main
import numpy as np

clf = joblib.load('all.pkl')

def make_prediction(image, label):
    # image = cv2.imread(image_path)
    image = np.array(image)
    print(image)
    predidted_label=main.PredictionModule(image)
    return predidted_label==int(label)

image_input = gr.Image(label="Upload Image", type="pil")
label_input = gr.Textbox(label="Enter Label")

# Define the output component
output = gr.Textbox(label="Prediction Result")

app = gr.Interface(fn=make_prediction, inputs=[image_input, label_input], outputs=output, title="Arabic Font Recognition")
app.launch(share=True)
