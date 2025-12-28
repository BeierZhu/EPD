import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import dnnlib
import tqdm

class InceptionFeatureExtractor:
    def __init__(self, device=torch.device('cuda')):
        self.device = device
        self.detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        self.detector_kwargs = dict(return_features=True)
        self.feature_dim = 2048
        
        with dnnlib.util.open_url(self.detector_url) as f:
            self.detector_net = pickle.load(f).to(device)
        self.detector_net.eval()

    def extract_features(self, images):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        images = images.to(self.device)
        
        features = self.detector_net(images, **self.detector_kwargs)
        return features

def compute_inception_mse_loss(student_images, teacher_images, feature_extractor):
    # Extract features
    student_features = feature_extractor.extract_features(student_images)
    teacher_features = feature_extractor.extract_features(teacher_images)
    
    # Compute MSE loss
    loss = F.mse_loss(student_features, teacher_features)

    return loss
