# Neural-Optimal-Transport-in-latent-space

fill it out later




## Download weigts:
Before you start working with the model, you need to download v1-5-pruned-emaonly.ckpt
file from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main and save it in the data directory.
So your folders should be like this:
   
    text_2_image(-main)/
    ├─ data/
    │  ├─ ldm_weights/
    │  │    ├─ v1-5-pruned-emaonly.ckpt
    │  │
    │  ├─ ...
    ├─ stable_diffusion/
    │  ├─ modules/
    │  ├─ ...
    ├─ dreambooth/ 
    │  ├─ dreambooth.py
    │  ├─  ...
    ...
    └── 
