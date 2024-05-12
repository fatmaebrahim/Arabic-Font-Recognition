
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, jsonify
from PIL import Image
import main
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    # Get the image path from the request
    # data = request.get_json()
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image'].stream.read()

    # Check if the file is empty
    # if file.filename == '':
    #     return 'No selected file'
    # print("///////////////elfile msh fady")
    # Save the uploaded file to a folder
    # print( file.filename)
    # image = cv2.imread(os.path.join('uploads', file.filename))

    image = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_COLOR)
    print(type(image))
    # img = Image.open(file)
    # img.save('2.jpg')
    # image=cv2.imread('2.jpg')
    print("/////////////////////////")

    label=main.PredictionModule(image)

    # Return the predicted label
    return jsonify({'label': str(label)})



if __name__ == '__main__':
    app.run(port=5003)