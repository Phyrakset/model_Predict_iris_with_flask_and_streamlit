from flask import Flask, request, jsonify
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = torch.load('model.pth')  # Ensure model.pth contains the full model
model.eval()  # Set the model to evaluation mode

# Define class-to-species mapping
class_to_species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

@app.route('/')
def home():
    return "Welcome to the Flask API for Predictions!", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data
        data = request.get_json()
        features = torch.tensor(data['data'], dtype=torch.float32)

        # Perform prediction
        with torch.no_grad():
            logits = model(features)  # Raw output from the model
            predicted_class = torch.argmax(logits, dim=1)  # Select class with the highest score

        # Map predicted class to species
        species = [class_to_species[int(cls)] for cls in predicted_class]

        # Return result
        return jsonify({"species": species})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
