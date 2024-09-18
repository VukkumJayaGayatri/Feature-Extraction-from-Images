import os
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import re
import easyocr
from tqdm import tqdm

# Define file paths
input_csv = r'C:\Users\Jaya Gayatri Vukkum\PycharmProjects\Amazon_ML\student_resource 3\dataset\Example.csv'
download_dir = r'C:\Users\Jaya Gayatri Vukkum\PycharmProjects\Amazon_ML\Download'
output_csv = r'C:\Users\Jaya Gayatri Vukkum\PycharmProjects\Amazon_ML\student_resource 3\dataset\test_sample.csv'

# Create the download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Function to download images
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
        else:
            print(f"Failed to retrieve image from {image_url}")
    except Exception as e:
        print(f"Error downloading image: {e}")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Define functions to extract entities
def extract_width(text):
    pattern = r'\b\d+(\.\d+)?\s*(centimetre|cm|foot|ft|inch|in|metre|m|millimetre|mm|yard|yd)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else None

# Repeat for other entity extraction functions like extract_depth, extract_height, etc.

entity_extraction_functions = {
    'width': extract_width,
    # Add other entity functions here (depth, height, weight, etc.)
}

def extract_entity_value(text, entity_name):
    extract_function = entity_extraction_functions.get(entity_name)
    if extract_function:
        return extract_function(text)
    return None

def convert_to_full_form(value, unit):
    units = {
        'cm': 'centimetre',
        # Add other unit mappings here
    }
    return f"{value} {units.get(unit.lower(), unit.lower())}"

# Read and process the dataset in chunks
chunksize = 100  # Number of rows per chunk
chunk_list = []

for chunk in pd.read_csv(input_csv, chunksize=chunksize):
    # Add a new column to store extracted entity values
    chunk['extracted_entity_value'] = ""

    for i, row in tqdm(chunk.iterrows(), total=chunk.shape[0]):
        image_url = row['image_link']
        save_path = os.path.join(download_dir, f"Image_{i}.jpg")
        download_image(image_url, save_path)

        # Use EasyOCR to extract text from the image
        results = reader.readtext(save_path)
        ocr_text = " ".join([result[1] for result in results])

        # Extract the entity value based on entity name
        entity_name = row['entity_name']
        entity_value = extract_entity_value(ocr_text, entity_name)

        # If no value is extracted, set it as an empty string
        if entity_value:
            match = re.match(r'(\d+(\.\d+)?)\s*(\D+)', entity_value)
            if match:
                value = match.group(1)
                unit = match.group(3).strip()
                entity_value = convert_to_full_form(float(value), unit)
            else:
                entity_value = ""
        chunk.at[i, 'extracted_entity_value'] = entity_value

    # Append processed chunk to list
    chunk_list.append(chunk)

# Concatenate all chunks and save to a CSV file
final_df = pd.concat(chunk_list)
final_df.to_csv(output_csv, index=False)

print("Entity values extracted and added to dataset successfully.")
