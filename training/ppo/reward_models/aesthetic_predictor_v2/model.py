import clip
import torch
import torch.nn as nn


def torch_normalized(a, axis=-1, order=2):
    l2 = torch.norm(a, dim=axis, p=order, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


class AestheticV2Model(nn.Module):
    def __init__(self, clip_path=None, predictor_path=None, device=None):
        super(AestheticV2Model, self).__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if clip_path:
            self.clip_encoder, self.preprocessor = clip.load("ViT-L/14", device=self.device, jit=False, download_root=clip_path)
        else:
            self.clip_encoder, self.preprocessor = clip.load("ViT-L/14", device=self.device, jit=False)

        state_dict = torch.load(predictor_path, map_location="cpu")
        modified_state_dict = {}
        for key in state_dict.keys():
            modified_state_dict[key[7:]] = state_dict[key]
        self.aesthetic_predictor = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        self.aesthetic_predictor.load_state_dict(modified_state_dict)
        self.aesthetic_predictor = self.aesthetic_predictor.to(self.device)

    def forward(self, x):
        batch = torch.stack([self.preprocessor(image) for image in x], dim=0).to(self.device)
        with torch.no_grad():
            batch = self.clip_encoder.encode_image(batch)
        batch = torch_normalized(batch).to(torch.float32)
        batch = self.aesthetic_predictor(batch)
        return batch


if __name__ == "__main__":
    model = AestheticV2Model(
        clip_path="/mnt/sda/models/clip/ViT-L-14.pt",
        predictor_path="/mnt/sda/models/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",
    ).cuda()
