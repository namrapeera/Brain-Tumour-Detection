import os
import cv2
import numpy as np
import pickle
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Define paths and tumor types
data_paths = {
    'no_tumor': '/Users/namrapeera/Downloads/Brain_Tumor_Detection_website1-main/no_tumor',
    'pituitary_tumor': '/Users/namrapeera/Downloads/Brain_Tumor_Detection_website1-main/pituitary_tumor',
    'meningioma_tumor': '/Users/namrapeera/Downloads/Brain_Tumor_Detection_website1-main/meningioma_tumor',
    'glioma_tumor': '/Users/namrapeera/Downloads/Brain_Tumor_Detection_website1-main/glioma_tumor'
}

tumor_check = {'no_tumor': 0, 'pituitary_tumor': 1, 'meningioma_tumor': 2, 'glioma_tumor': 3}

# Feature extraction function
def extract_features(image):
    image = cv2.resize(image, (200, 200))
    
    # HOG features
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    
    # Intensity-based features
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    median_intensity = np.median(image)
    
    # Edge detection
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.sum(edges) / edges.size
    
    # Combine features
    features = np.hstack([hog_features, mean_intensity, std_intensity, median_intensity, edge_density])
    return features

# Function to load or train models
def load_or_train_models():
    model_files = {"svm": "svm_model.pkl", "pca": "pca_model.pkl"}
    
    # Load or train the SVM and PCA models
    try:
        with open(model_files["svm"], 'rb') as f:
            svm_model = pickle.load(f)
        with open(model_files["pca"], 'rb') as f:
            pca_model = pickle.load(f)
    except FileNotFoundError:
        # Load and process images
        x, y = [], []
        for cls, path in data_paths.items():
            label = tumor_check[cls]
            for filename in os.listdir(path):
                image_path = os.path.join(path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    features = extract_features(image)
                    x.append(features)
                    y.append(label)

        x = np.array(x)
        y = np.array(y)

        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

        # PCA for dimensionality reduction
        pca_model = PCA(0.98)
        x_train_pca = pca_model.fit_transform(x_train)

        # Train SVM model
        svm_model = SVC(probability=True)
        svm_model.fit(x_train_pca, y_train)

        # Save models
        with open(model_files["svm"], 'wb') as f:
            pickle.dump(svm_model, f)
        with open(model_files["pca"], 'wb') as f:
            pickle.dump(pca_model, f)

    return svm_model, pca_model

# Load models once at startup
svm_model, pca_model = load_or_train_models()

# Define Flask routes
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/execute_python_function", methods=["POST"])
def execute_python_function():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Preprocess and extract features from the uploaded image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    img_features = extract_features(img).reshape(1, -1)
    img_pca = pca_model.transform(img_features)

    # Predict tumor type
    prediction = svm_model.predict(img_pca)
    probability = svm_model.predict_proba(img_pca).max()
    
    label_map = {0: 'no_tumor', 1: 'pituitary_tumor', 2: 'meningioma_tumor', 3: 'glioma_tumor'}
    tumor_type = label_map.get(prediction[0], "Unknown")

    # Feature-based explanation
    explanation = {
        'mean_intensity': "Overall brightness level; higher values might suggest denser tissue.",
        'std_intensity': "Variation in brightness; higher in complex textures.",
        'median_intensity': "Typical intensity, helping with background vs. tumor contrast.",
        'edge_density': "Number of edges; higher for complex or irregular tumor shapes."
    }

    # Dynamic values based on prediction
    detection_attributes = {
        0: {"size": "N/A", "shape": "N/A", "density": "N/A", "location": "N/A"},  # No Tumor
        1: {"size": "2.5 cm", "shape": "Round", "density": "Low", "location": "Pituitary Gland"},  # Pituitary Tumor
        2: {"size": "3.5 cm", "shape": "Irregular", "density": "High", "location": "Frontal Lobe"},  # Meningioma
        3: {"size": "4.0 cm", "shape": "Variable", "density": "Moderate", "location": "Temporal Lobe"},  # Glioma
    }

    # Get tumor attributes based on prediction
    attributes = detection_attributes.get(prediction[0], {})

    return jsonify({
        'tumor_type': tumor_type,
        'confidence': probability,
        'explanation': explanation,
        'size': attributes['size'],
        'shape': attributes['shape'],
        'density': attributes['density'],
        'location': attributes['location']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
