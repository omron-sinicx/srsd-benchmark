from torch import nn


def build_decoder(decoder_config):
    base_layer = nn.TransformerDecoderLayer(**decoder_config['layer'])
    norm_layer = None
    norm_config = decoder_config.get('norm', None)
    if norm_config is not None:
        norm_layer = nn.__dict__[decoder_config['norm']]() if 'norm' in decoder_config else None
    return nn.TransformerDecoder(base_layer, norm=norm_layer, **decoder_config['kwargs'])
