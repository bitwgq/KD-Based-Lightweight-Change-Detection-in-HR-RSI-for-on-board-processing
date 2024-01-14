from typing import Optional, Union, List
from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead


class FC_Siam(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        siam_encoder: bool = True,
        fusion_form: str = "concat",
        **kwargs
    ):
        super().__init__()

        self.siam_encoder = siam_encoder

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if not self.siam_encoder:
            self.encoder_non_siam = get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            fusion_form=fusion_form,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

if __name__ == "__main__":
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input1 = torch.randn(1, 3, 256, 256).to(device)
    input2 = torch.randn(1, 3, 256, 256).to(device)
    net = FC_Siam()
    res = net.forward(input1, input2)
    print(res.shape)
