from .encoder import VAE_Encoder
from .decoder import VAE_Decoder
from .converter import get_model_weight_dict

def load_models_from_standard_weights(path, device):
    state_dict = get_model_weight_dict(path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    return {
        'encoder': encoder,
        'decoder': decoder,
    }