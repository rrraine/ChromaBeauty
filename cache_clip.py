from transformers import CLIPTextModel, CLIPTokenizer
CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP cached!")