import cv2
import numpy as np
import hashlib
import json
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# --- Step 1: Image Acquisition ---
def acquire_image(image_path):
    print(f"[*] Step 1: Acquiring image from '{image_path}'...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Could not load image from path: {image_path}")
    else:
        print("[+] Image acquired successfully.")
    return image


# --- New Step: Visually "Detect" Pneumonia Area (Demonstration) ---
def detect_pneumonia_area(image):
    """
    DISCLAIMER: This is a simplified visual demonstration, NOT a medically accurate detection.
    It highlights the brightest region of the image, which can sometimes correspond to
    areas of interest in a chest X-ray.
    """
    print("\n[*] VISUAL DEMO: Attempting to highlight potential area of interest...")
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to reduce noise and improve thresholding
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Apply a binary threshold to isolate the brightest areas
    # All pixel values above 200 will be set to 255 (white)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour by area
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding box coordinates for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw a red rectangle around the detected area on the original image
        output_image = image.copy()
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output_image, "Potential Area", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        print("[+] Highlighted the largest bright region. Displaying image.")
        return output_image
    else:
        print("[-] No significant bright regions found to highlight.")
        return image


# --- Step 2: Preprocessing & Feature Extraction ---
def preprocess_and_extract_features(image):
    print("\n[*] Step 2: Preprocessing image and extracting features...")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (128, 128))
    features = resized_image.flatten()
    print("[+] Preprocessing and feature extraction complete.")
    return features


# --- Step 3: Train Classifier and Evaluate with Metrics ---
def train_and_evaluate_classifier():
    """
    Trains an SVM classifier on a synthetic dataset and evaluates its performance.
    """
    print("\n[*] Step 3: Training and evaluating the diagnostic model...")
    
    # 1. Create a more realistic synthetic dataset
    # 1000 samples, 128*128 features each, 2 classes (Normal/Pneumonia)
    n_features = 128 * 128
    X, y = make_classification(n_samples=1000, n_features=n_features, n_informative=10, n_redundant=5, n_classes=2, random_state=42)

    # 2. Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[.] Dataset split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # 3. Train the SVM Classifier
    print("[.] Training SVM classifier...")
    classifier = svm.SVC(kernel='linear', random_state=42)
    classifier.fit(X_train, y_train)
    print("[+] Classifier trained successfully.")
    
    # 4. Make predictions on the unseen test data
    y_pred = classifier.predict(X_test)

    # 5. Calculate and display evaluation metrics
    print("\n--- Model Evaluation Metrics ---")
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    print(f"  - Accuracy:  {accuracy:.2f}%")
    print(f"  - Precision: {precision:.2f}%")
    
    # 6. Generate and plot the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n  - Confusion Matrix:")
    print(f"    True Negatives: {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f"    False Negatives: {cm[1][0]} | True Positives:  {cm[1][1]}")
    
    # Plotting the matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    # Use plt.show(block=False) or save the figure instead of plt.show()
    # to avoid blocking the script, or just show it at the end.
    # For this script, plt.show() here is fine.
    plt.show()

    return classifier


# --- Step 4: Data Hashing & Blockchain Transaction Preparation ---
def prepare_blockchain_transaction(patient_id, diagnostic_data):
    print("\n[*] Step 4: Preparing data for blockchain...")
    payload = { "patient_id": patient_id, "timestamp": datetime.utcnow().isoformat() + "Z", "analysis_results": diagnostic_data }
    payload_string = json.dumps(payload, sort_keys=True)
    data_hash = hashlib.sha256(payload_string.encode()).hexdigest()
    # THIS IS THE CORRECTED LINE:
    transaction = { "transaction_id": hashlib.sha256(str(datetime.now()).encode()).hexdigest(), "data_payload": payload, "data_hash": data_hash }
    print(f"[.] Data Hash (SHA-256): {data_hash}")
    print("[+] Blockchain transaction prepared successfully.")
    return transaction


# --- Main Workflow ---
if __name__ == "__main__":
    
    # --- THIS IS THE MODIFIED LINE ---
    # It now looks for the image in the same folder as the script.
    IMAGE_FILE_PATH = "chestimage.jpg"
    # -----------------------------------

    # --- Execute Model Training and Evaluation ---
    trained_model = train_and_evaluate_classifier()
    
    # --- Run Pipeline on Your Specific Image ---
    patient_image = acquire_image(IMAGE_FILE_PATH)
    
    if patient_image is not None:
        # Show the "detected" area
        highlighted_image = detect_pneumonia_area(patient_image)
        cv2.imshow("Pneumonia Detection Demo (Not for Diagnosis)", highlighted_image)
        cv2.waitKey(0) # Press any key to close the image window
        cv2.destroyAllWindows()

        # Extract features and make a final prediction for your image
        features = preprocess_and_extract_features(patient_image)
        prediction_code = trained_model.predict(features.reshape(1, -1))[0]
        prediction_label = "Pneumonia" if prediction_code == 1 else "Normal"
        
        print(f"\n--- Prediction for {IMAGE_FILE_PATH} ---")
        print(f"Final Diagnosis: {prediction_label}")
        
        diagnostic_result = {"diagnosis": prediction_label}

        # Prepare the blockchain transaction
        blockchain_transaction = prepare_blockchain_transaction(
            patient_id="PID-12345",
            diagnostic_data=diagnostic_result
        )
        
        print("\n" + "="*50)
        print("          FINAL BLOCKCHAIN TRANSACTION")
        print("="*50)
        print(json.dumps(blockchain_transaction, indent=4))
        print("="*50)
