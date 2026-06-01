import torch
from terratorch.models import EncoderDecoderFactory

def load_model(model_path):
    factory = EncoderDecoderFactory()
    model   = factory.build_model(
        task                = "segmentation",
        backbone            = "prithvi_eo_v2_300_tl",
        backbone_pretrained = False,
        backbone_num_frames = 1,
        backbone_bands      = ["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"],
        num_classes         = 4,
        decoder             = "UperNetDecoder",
    ).cuda()

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model