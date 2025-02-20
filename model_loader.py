import torch
import torchvision.models.segmentation as models

def load_segmentation_model():
    """Carga un modelo de segmentación preentrenado (DeepLabV3)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = models.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval()  # Modo evaluación para inferencia
    
    return model