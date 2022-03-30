import torch.nn as nn
import torch


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, use_dropout):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, True)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding
                                   layer: reflect | replicate | zero
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer,
                              and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                        is not implemented"
            )

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                      is not implemented"
            )
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Resnet(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(Resnet, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=0, bias=True),
            nn.ReLU(True),
        ]

        n_blocks = 9
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    nf,
                    padding_type='reflect',
                    use_dropout=False,
                )
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, output_nc, kernel_size=7, padding=0, bias=True),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        res = self.model(x.type(torch.cuda.FloatTensor))
        return res
