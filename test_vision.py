import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from app import analyze_image_with_vision, parse_vision_response

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')
img.save('dummy.jpg')

vision_text = analyze_image_with_vision(Image.open('dummy.jpg'))
print("RAW VISION TEXT:")
print(vision_text)

print("---")
image_description, extracted_text, is_social_media, image_warning = parse_vision_response(vision_text)
print("PARSED:")
print("desc:", repr(image_description))
print("text:", repr(extracted_text))
print("social:", is_social_media)
