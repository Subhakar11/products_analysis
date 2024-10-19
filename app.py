from flask import Flask, render_template, request, send_file, redirect, url_for
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import re
import openpyxl
import os

app = Flask(__name__)

# Load the Qwen2 model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Set directory for uploads
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Regular expression patterns to extract data
packaged_product_pattern = r"Product Name: (.*)\n  - Product Category: (.*)\n  - Product Quantity: (.*)\n  - Product Count: (.*)\n  - Expiry Date: (.*)"
fruits_vegetables_pattern = r"Type of fruit/vegetable: (.*)\n  - Freshness Index: (.*)\n  - Estimated Shelf Life: (.*)"

@app.route('/')
def home():
    return render_template('index.html', analysis_result=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request.")  # Debugging line
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        print("No file selected.")  # Debugging line
        return redirect(url_for('home'))
    
    if file:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"File saved at {filepath}")  # Debugging line
        
        # Process the image and generate output text
        output_text = process_image(filepath)
        
        if "Error" in output_text:
            return "An error occurred during image processing."

        # Generate Excel report using the output text
        generate_excel(output_text)
        
        # Return the analysis result on the web page
        return render_template('index.html', analysis_result=output_text)

def process_image(image_path):
    try:
        # Open the image using PIL
        image = Image.open(image_path).resize((512, 512))
        
        # Prepare text prompt for predicting product details and freshness
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_url": "Captured from uploaded file"
                    },
                    {
                        "type": "text",
                        "text": """This image contains fruits, vegetables, or packaged products.
                        Please analyze the image and provide:
                        - For packaged products:
                            - Product Name
                            - Product Category
                            - Product Quantity
                            - Product Count
                            - Expiry Date (if available)
                        - For fruits and vegetables:
                            - Type of fruit/vegetable
                            - Freshness Index (based on visual cues)
                            - Estimated Shelf Life"""
                    }
                ]
            }
        ]

        # Prepare inputs for the model
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate output
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        print("Generated Output:", output_text)  # Debugging line
        return output_text

    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error during processing"

def generate_excel(output_text):
    # Create a new Excel workbook
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Product Analysis"
    headers = ["Product Name", "Category", "Quantity", "Count", "Expiry Date", "Freshness Index", "Shelf Life"]
    sheet.append(headers)

    # Extract packaged product information
    packaged_product_match = re.search(packaged_product_pattern, output_text)
    fruits_vegetables_match = re.search(fruits_vegetables_pattern, output_text)

    if packaged_product_match:
        product_name = packaged_product_match.group(1).strip()
        category = packaged_product_match.group(2).strip()
        quantity = packaged_product_match.group(3).strip()
        count = packaged_product_match.group(4).strip()
        expiry_date = packaged_product_match.group(5).strip()
        freshness_index = shelf_life = "N/A"
    elif fruits_vegetables_match:
        product_name = fruits_vegetables_match.group(1).strip()
        category = "Fruit/Vegetable"
        freshness_index = fruits_vegetables_match.group(2).strip()
        shelf_life = fruits_vegetables_match.group(3).strip()
        quantity = count = expiry_date = "N/A"
    else:
        product_name = category = quantity = count = expiry_date = freshness_index = shelf_life = "N/A"

    # Insert data into Excel
    sheet.append([product_name, category, quantity, count, expiry_date, freshness_index, shelf_life])

    # Save Excel file
    excel_path = os.path.join(UPLOAD_FOLDER, 'product_analysis.xlsx')
    workbook.save(excel_path)
    print(f"Excel file saved at {excel_path}")  # Debugging line
    return excel_path

if __name__ == '__main__':
    app.run(debug=False)
