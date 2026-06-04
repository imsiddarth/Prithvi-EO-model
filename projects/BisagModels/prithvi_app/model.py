import torch
from terratorch.models import EncoderDecoderFactory

def load_model(model_path, num_classes=4, is_segmentation=True):
    factory = EncoderDecoderFactory()
    model   = factory.build_model(
        task                = "segmentation" if is_segmentation else "classification",
        backbone            = "prithvi_eo_v2_300_tl",
        backbone_pretrained = False,
        backbone_num_frames = 1,
        backbone_bands      = ["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"],
        num_classes         = num_classes,
        decoder             = "UperNetDecoder" if is_segmentation else "IdentityDecoder",
    ).cuda()

    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()
    return model