import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3


def calculate_inception_score(images, batch_size: int = 32, splits: int = 10):
    # 이미지를 Inception 모델에 전달하여 특징 벡터를 추출
    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
    ).eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize(299, antialias=True),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    activations = []
    data_loader = DataLoader(images, batch_size=batch_size)
    with torch.no_grad():
        for batch in data_loader:
            batch = preprocess(batch)
            features = model(batch)
            activations.append(features)

    activations = torch.cat(activations, dim=0)

    # 이미지의 Inception Score 계산
    scores = []
    for i in range(splits):
        subset = activations[
            i
            * (activations.size(0) // splits) : (i + 1)
            * (activations.size(0) // splits),
            :,
        ]
        p_yx = F.softmax(subset, dim=1).mean(dim=0)
        kl_d = subset * (torch.log(subset) - torch.log(torch.unsqueeze(p_yx, 0)))
        kl_d = torch.mean(torch.sum(kl_d, dim=1))
        scores.append(torch.exp(kl_d.item()))

    # Inception Score의 평균과 표준 편차 계산
    is_mean = torch.mean(torch.tensor(scores))
    is_std = torch.std(torch.tensor(scores))

    return is_mean.item(), is_std.item()


def calculate_fid(real_images, generated_images, batch_size=32):
    # 실제 이미지의 특징 추출
    model = (
        inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        .eval()
        .cuda()
    )
    preprocess = transforms.Compose(
        [
            transforms.Resize(299, antialias=True),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    real_activations = []
    real_loader = DataLoader(real_images, batch_size=batch_size)
    with torch.no_grad():
        for batch in real_loader:
            batch = preprocess(batch).cuda()
            features = model(batch)
            real_activations.append(features)

    real_activations = torch.cat(real_activations, dim=0)

    # 생성된 이미지의 특징 추출
    generated_activations = []
    generated_loader = DataLoader(generated_images, batch_size=batch_size)
    with torch.no_grad():
        for batch in generated_loader:
            batch = preprocess(batch).cuda()
            features = model(batch)
            generated_activations.append(features)

    generated_activations = torch.cat(generated_activations, dim=0)

    if generated_activations.shape[0] == 0:  # 예외처리
        return 1e5
    # 실제 이미지와 생성된 이미지의 특징 통계 계산
    mu_real = torch.mean(real_activations, dim=0)
    mu_generated = torch.mean(generated_activations, dim=0)
    sigma_real = (
        torch.matmul((real_activations - mu_real).T, real_activations - mu_real)
        / real_activations.shape[0]
    )
    sigma_generated = (
        torch.matmul(
            (generated_activations - mu_generated).T,
            generated_activations - mu_generated,
        )
        / generated_activations.shape[0]
    )

    # FID 계산
    diff = mu_real - mu_generated
    cov_mean = sqrtm(sigma_real.cpu().numpy() @ sigma_generated.cpu().numpy())
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid_score = torch.norm(diff) + torch.trace(
        sigma_real.cpu() + sigma_generated.cpu() - 2 * cov_mean
    )

    return fid_score.item()
