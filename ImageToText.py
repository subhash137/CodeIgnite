# import requests
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# # img_url = 'snapshots/snapshot_72.jpg' 
# img_url = 'image.png' 
# raw_image = Image.open(img_url).convert('RGB')

# # conditional image captioning
# text = "A accident of "
# inputs = processor(raw_image, text, return_tensors="pt")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))



from vertexai.generative_models import GenerativeModel, Image
import json
vision_model = GenerativeModel("gemini-pro-vision")

# Local image
image = Image.load_from_file("image.png")
output = vision_model.generate_content(["describe about the accident in image?", image])

# data = json.loads(output)

# Extract the text from the parts
text = output.candidates[0].content.parts[0].text

print(text)
#What is the accident shown in this image and are there any  victims, injured people , 