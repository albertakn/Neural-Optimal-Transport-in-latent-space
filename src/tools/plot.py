from matplotlib import pyplot as plt


def plot_original(batch):
    fig, axes = plt.subplots(1, 5, figsize=(10, 1), dpi=100)
    batch = rescale(batch.clone(), (-1, 1), (0, 255), clamp=True)
    batch = batch.to("cpu", torch.uint8)
    for i in range(5):
        axes[i].imshow(batch[i].permute(1,2,0))
        axes[i].set_xticks([]); axes[i].set_yticks([])
    fig.tight_layout(pad=0.1)


def plot_many_images(multibatch):
    fig, axes = plt.subplots(4, 5, figsize=(10, 4), dpi=100)
    for i in range(5):
        with torch.no_grad():
            decoded_batch = models['decoder'](multibatch[i].to(DEVICE)).detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()

        decoded_batch = rescale(decoded_batch, (-1, 1), (0, 255), clamp=True)
        decoded_batch = decoded_batch.to("cpu", torch.uint8)

        for j in range(4):
            axes[j, i].imshow(decoded_batch[j].mul(0.5).add(0.5).clip(0,1).permute((1,2,0)))
            axes[j, i].set_xticks([]); axes[j, i].set_yticks([])
    fig.tight_layout(pad=0.1)