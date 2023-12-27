# Neural-Optimal-Transport-in-latent-space

Neural Optimal Transport (NOT) emerges as a promising solution for tackling unpaired style transfer challenges. This study demonstrates the application of the NOT approach on realistic and anime human faces. However, addressing high-resolution style transfer demands substantial computational resources. To circumvent this, we leverage Stable Diffusion auto-encoders, conducting experiments within latent spaces to efficiently solve the NOT problem. Our findings not only underscore the potential of NOT in the realm of unpaired style transfer but also present a computationally efficient strategy for handling high-resolution scenarios.



## Download weigts:
Before you start working with the model, you need to download v1-5-pruned-emaonly.ckpt
file from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main and save it in the data directory.
So your folders should be like this:
   
    text_2_image(-main)/
    ├─ data/
    │  ├─ ldm_weights/
    │  │  ├─ v1-5-pruned-emaonly.ckpt
    │  │
    │  ├─ ...
    ├─ notebooks/
    │  ├─ NOT_in_latent_space_training_example.ipynb
    │  ├─ ...
    ├─ src/ 
    │  ├─ vae/
    │  │  ├─ ...
    │  ├─  ...
    ...
    └── 

NOT model weights (should be placed in data/NOT_model_weights/):

   64x64:
   
      https://disk.yandex.ru/d/YJqMaJlXg1cYRw
      
      https://disk.yandex.ru/d/BJCgRZLIm9dfwQ
      
   vae latent:
   
      https://disk.yandex.ru/d/_OMZ0-JMZqJaZg
      
      https://disk.yandex.ru/d/qn9P1pN8Zd3J7Q
      
datasets (should be placed in data/)

   256x256:
   
      source - https://disk.yandex.ru/d/Z_EOL0dUuxJYug
      
      target - https://disk.yandex.ru/d/mOQXlEHNkSk8JQ

      
## How to use:
download weights as in previous paragraph and see *.ipynb files in notebooks/ 



## References:
Our work is heavily on https://github.com/iamalexkorotin/NeuralOptimalTransport/tree/main
