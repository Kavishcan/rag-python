from pdf2image import convert_from_path
import pytesseract

# Convert PDF pages to images
images = convert_from_path('data/AL-ICT-2023-P1.pdf')

# Extract text from each image
text = ""
for image in images:
    text += pytesseract.image_to_string(image)

print(text)
