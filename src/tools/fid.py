import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy


def predict(
    model: pl.LightningModule,
    data: torch.Tensor,
    batch_size: int,
    verbose: bool = True,
):
    output = []
    slicer = range(0, len(data), batch_size)
    if verbose:
        slicer = tqdm.tqdm(slicer)

    model.eval()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    with torch.no_grad():
        for i in slicer:
            x = data[i : i + batch_size]
            if use_gpu:
                x = x.cuda()
            y = model(x)
            if use_gpu:
                y = y.cpu()
            output.append(y)
    if use_gpu:
        model.cpu()

    output = torch.cat(output, dim=0)
    return output


class InceptionHeadless(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.model = torch.hub.load("pytorch/vision", "inception_v3", pretrained=True)
        # remove last fc layer
        self.model.fc = nn.Identity()

    def forward(self, z):
        x = z.permute(0, 3, 1, 2)
        x = self.upsample(x)
        return self.model(x)


def calculate_activations(data, inception, batch_size=32, verbose=False):
    # Calculate activations of Pool3 layer of InceptionV3
    if verbose:
        print("Calculating activations...")
    activations = predict(inception, data, batch_size=32)
    return activations


def calculate_activation_statistics(activations):
    # Calculate mean and covariance of activations. Mind the dimensions!
    mu = activations.mean(dim=0)
    t = activations - mu
    sigma = torch.Tensor(np.cov(activations, rowvar = False))
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    sigma1_sigma2 = scipy.linalg.sqrtm(np.dot(sigma1, sigma2))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(sigma1_sigma2):
        sigma1_sigma2 = sigma1_sigma2.real

    # Product might be almost singular
    if not np.isfinite(sigma1_sigma2).all():
        offset = np.eye(sigma1.shape[0]) * eps
        sigma1_sigma2 = scipy.linalg.sqrtm(np.dot(sigma1 + offset, sigma2 + offset))

    diff = mu1 - mu2

    # use diff, sigma1, sigma2 to calculate FID according to the formula above
    return (torch.sum(diff**2) + torch.sum(torch.diag(sigma1 + sigma2 - 2*sigma1_sigma2))).item()


def calculate_fid_score(real_data, fake_data, verbose=False):
    # Run inception on real and fake data to obtain activations
    inception = InceptionHeadless()
    inception.eval()

    real_activations = calculate_activations(real_data, inception)
    fake_activations = calculate_activations(fake_data, inception)

    # Calculate mu and sigma for both real and fake activations
    real_mu, real_sigma = calculate_activation_statistics(real_activations)
    fake_mu, fake_sigma = calculate_activation_statistics(fake_activations)

    # Calculate Frechet distance
    return calculate_frechet_distance(
        real_mu,
        real_sigma,
        fake_mu,
        fake_sigma,
    )