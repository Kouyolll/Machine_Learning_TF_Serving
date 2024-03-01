from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import time 
import os 
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'mnist_model.h5'), compile=False)

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>MNIST Digit Recognition</title>
        <style>
            body {
                background-color: #1E90FF;
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                text-align: center;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background-color: #007ACC;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                max-width: 500px;
            }
            h1 {
                font-size: 2em;
            }
            #canvas {
                border: 1px solid black;
                background-color: white;
                cursor: crosshair;
            }
            .button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
                margin-top: 10px;
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MNIST Digit Recognition</h1>
            <canvas id="canvas" width="280" height="280"></canvas><br>
            <button class="button" onclick="clearCanvas()">Clear</button>
            <button class="button" onclick="submitDrawing()">Upload and Predict</button>
            <p id="result"></p>
        </div>
        <script>
            var canvas = document.getElementById('canvas');
            var ctx = canvas.getContext('2d');
            ctx.fillStyle = "white";  // White background
            ctx.fillRect(0, 0, canvas.width, canvas.height);  // Fill canvas with white
            var drawing = false;
            var mousePos = { x:0, y:0 };
            var lastPos = mousePos;

            canvas.addEventListener("mousedown", function (e) {
                drawing = true;
                lastPos = getMousePos(canvas, e);
            }, false);

            canvas.addEventListener("mouseup", function (e) {
                drawing = false;
                ctx.beginPath(); // Start a new path after lifting the mouse up
            }, false);

            canvas.addEventListener("mousemove", function (e) {
                mousePos = getMousePos(canvas, e);
            }, false);

            function getMousePos(canvasDom, mouseEvent) {
                var rect = canvasDom.getBoundingClientRect();
                return {
                    x: mouseEvent.clientX - rect.left,
                    y: mouseEvent.clientY - rect.top
                };
            }

            function renderCanvas() {
                if (drawing) {
                    ctx.lineWidth = 20;
                    ctx.lineCap = 'round';
                    ctx.strokeStyle = 'black'; // Draw in black
                    ctx.moveTo(lastPos.x, lastPos.y);
                    ctx.lineTo(mousePos.x, mousePos.y);
                    ctx.stroke();
                    lastPos = mousePos;
                }
            }

            (function drawLoop() {
                requestAnimationFrame(drawLoop);
                renderCanvas();
            })();

            function clearCanvas() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "white";  // Reset to white background after clearing
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.beginPath();
            }

            function getInvertedImageDataURL(canvas) {
                var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                var data = imageData.data;
                for (var i = 0; i < data.length; i += 4) {
                    // Invert the colors: black to white and white to black
                    data[i] = 255 - data[i];       // Red
                    data[i + 1] = 255 - data[i + 1]; // Green
                    data[i + 2] = 255 - data[i + 2]; // Blue
                }
                ctx.putImageData(imageData, 0, 0);
                var dataURL = canvas.toDataURL('image/png');
                // Restore the original drawing after taking the snapshot
                ctx.putImageData(imageData, 0, 0);
                return dataURL;
            }

            function submitDrawing() {
                var startTime = new Date();  // Start the timer

                var dataURL = getInvertedImageDataURL(canvas);
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ image: dataURL })
                }).then(response => response.json())
                .then(data => {
                    var endTime = new Date();  // End the timer
                    var totalDuration = (endTime - startTime) / 1000;  // Total duration in seconds

                    document.getElementById('result').innerHTML = 
                        'Predicted Digit: ' + data.digit + '<br>' +
                        'Prediction Time: ' + data.duration.toFixed(5) + ' seconds' + '<br>' +
                        'Total Time: ' + totalDuration.toFixed(5) + ' seconds';
                });
            }
        </script>

    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    header, encoded = image_data.split(",", 1)
    binary = base64.b64decode(encoded)

    image = Image.open(io.BytesIO(binary)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0

    # Start the timer
    start_time = time.time()

    # Perform the prediction
    prediction = model.predict(image)
    digit = np.argmax(prediction)

    # End the timer
    end_time = time.time()
    duration = end_time - start_time  # Duration in seconds

    return jsonify({'digit': int(digit), 'duration': duration})

if __name__ == '__main__':
    app.run(debug=True)
