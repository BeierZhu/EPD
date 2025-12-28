# import
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor


# load model
class PickScoreModel(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        processor_name_or_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_pretrained_name_or_path: str = "yuvalkirstain/PickScore_v1",
    ) -> None:
        super().__init__()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    def forward(self, prompt, images):
        # Preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # Embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # Score
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

            # Get probabilities if you have multiple images to choose from
            # probs = torch.softmax(scores, dim=-1)

        return scores.cpu().tolist()


if __name__ == "__main__":
    model = PickScoreModel(
        device="cuda",
        processor_name_or_path="/mnt/sda/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_pretrained_name_or_path="/mnt/sda/models/yuvalkirstain/PickScore_v1",
    )
    pil_images = [Image.open("demo/misaka.png")]
    prompt = "fantastic, increadible prompt"
    print(model.calc_probs(prompt, pil_images))
