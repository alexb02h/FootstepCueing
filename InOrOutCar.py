from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch, cv2

class InCarVisualVerifier :
    def __init__(self) :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.labels = ["a person sitting inside a car cabin", "a person standing outside on the street"]
        
    @torch.no_grad()
    def verify(self, person_crop) : 
        if person_crop is None or len(person_crop.shape) < 2 or person_crop.shape[0] < 20 or person_crop.shape[1] < 20 : return "Unknown"
        
        image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        if probs[0] > 0.65 : return "Inside"
        else : return "Outside"