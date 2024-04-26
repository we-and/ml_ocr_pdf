import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
import io

# Initialize Tesseract-OCR; specify the path if necessary
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Update the path based on your installation
def preprocess_image(image):
    """Pre-process the image to improve OCR accuracy."""
    # Check if image is grayscale; convert to BGR if it is
    if len(np.array(image).shape) == 2:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # Apply thresholding, noise removal, etc.
    processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(processed_image)

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of the PDF."""
    doc = fitz.open(pdf_path)
    for page in doc:
        # Extract image from each page
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            preprocessed_image = preprocess_image(image)
            text = pytesseract.image_to_string(preprocessed_image)
            print(f"Text from page {page.number}, image {img_index}:\n{text}\n")

# Example usage
extract_text_from_pdf("examples/scan.pdf")