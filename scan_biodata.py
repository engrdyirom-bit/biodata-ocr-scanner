import os
import cv2
import pytesseract
import json
import requests
import numpy as np

# Set Tesseract config for block text
CUSTOM_CONFIG = r'--oem 3 --psm 6'

def get_latest_scan():
    """Finds the most recent image uploaded in the scans/ directory."""
    scan_dir = "scans"
    if not os.path.exists(scan_dir):
        os.makedirs(scan_dir)
        return None
    
    files = [os.path.join(scan_dir, f) for f in os.listdir(scan_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        return None
    
    # Return the latest modified file
    return max(files, key=os.path.getmtime)

def align_document(image_path):
    """Aligns paper using 4 corner fiducial markers."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Target size (Standard A4 ratio in pixels: 1240 x 1754)
    target_w, target_h = 1240, 1754
    aligned = cv2.resize(image, (target_w, target_h))
    return aligned

def extract_ocr_fields(image):
    """Crops pre-defined relative coordinate bounding boxes and extracts handwritten/typed text."""
    h, w, _ = image.shape
    
    # Coordinates normalized as percentages (ymin, xmin, ymax, xmax) for bounding boxes
    field_coords = {
        "last_name":          (0.125, 0.120, 0.145, 0.380),
        "first_name":         (0.125, 0.390, 0.145, 0.650),
        "middle_name":        (0.125, 0.660, 0.145, 0.900),
        "dob":                (0.155, 0.120, 0.175, 0.380),
        "pob":                (0.155, 0.390, 0.175, 0.650),
        "age":                (0.155, 0.660, 0.175, 0.750),
        "gender":             (0.155, 0.760, 0.175, 0.900),
        "civil_status":       (0.185, 0.120, 0.205, 0.380),
        "citizenship":        (0.185, 0.390, 0.205, 0.650),
        "religion":           (0.185, 0.660, 0.205, 0.900),
        "address":            (0.215, 0.120, 0.235, 0.900),
        "provincial_address": (0.245, 0.120, 0.265, 0.900),
        "spouse":             (0.300, 0.120, 0.320, 0.500),
        "father":             (0.330, 0.120, 0.350, 0.500),
        "mother":             (0.360, 0.120, 0.380, 0.500),
        "emerg_person":       (0.390, 0.120, 0.410, 0.450),
        "emerg_contact":      (0.390, 0.650, 0.410, 0.900)
    }

    results = {}
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for field, (ymin, xmin, ymax, xmax) in field_coords.items():
        # Convert percentages to actual pixel coordinates
        y1, y2 = int(ymin * h), int(ymax * h)
        x1, x2 = int(xmin * w), int(xmax * w)
        
        crop = gray_img[y1:y2, x1:x2]
        
        # Image preprocessing for clean OCR
        crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Extract text via Tesseract OCR
        text = pytesseract.image_to_string(crop, config=CUSTOM_CONFIG)
        results[field] = text.strip()

    return results

def send_to_google_sheets(data_payload):
    """Posts extracted data payload to Google Apps Script Webhook."""
    webhook_url = os.environ.get("GOOGLE_SHEET_WEBHOOK_URL")
    if not webhook_url:
        print("ERROR: GOOGLE_SHEET_WEBHOOK_URL environment variable is missing.")
        return

    headers = {'Content-Type': 'application/json'}
    response = requests.post(webhook_url, data=json.dumps(data_payload), headers=headers)
    print("Google Sheets Response:", response.text)

if __name__ == "__main__":
    latest_scan = get_latest_scan()
    if not latest_scan:
        print("No image found in scans/ directory to process.")
    else:
        print(f"Processing scan: {latest_scan}")
        aligned_img = align_document(latest_scan)
        extracted_data = extract_ocr_fields(aligned_img)
        print("Extracted Data:", json.dumps(extracted_data, indent=2))
        send_to_google_sheets(extracted_data)