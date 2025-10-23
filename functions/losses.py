# ================================================================
# functions/losses.py
# Diffusion 학습 손실 (BB → T1CE)
# ================================================================

import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          x_c: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor):
    """
    Diffusion 모델 학습용 노이즈 예측 손실 함수

    Args:
        model (torch.nn.Module): 학습 중인 Diffusion 모델
        x0 (torch.Tensor): 원본 깨끗한 이미지 (target 이미지, 예: T1CE)
        x_c (torch.Tensor): 조건부 입력 이미지 (예: BB)
        t (torch.LongTensor): timestep (diffusion 단계)
        e (torch.Tensor): 랜덤 노이즈 (가우시안 분포에서 샘플링)
        b (torch.Tensor): 베타 스케줄 (beta_t 값, 길이=T)

    Returns:
        torch.Tensor: 예측 노이즈와 실제 노이즈의 MSE 손실 값
    """

    # alpha_t 계산 : 각 t 단계에서 남아있는 "signal 비율"
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

    # 깨끗한 이미지 x0에 노이즈 e를 추가한 버전 생성
    xt = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    # [noisy 이미지 xt, 조건 이미지 x_c]를 채널 방향으로 합침
    x = torch.cat([xt, x_c], dim=1)

    # noisy 이미지 xt와 timestep t를 입력받아 각 픽셀의 노이즈 값을 예측하도록 학습됨
    output, _ = model(x, t.float())

    # 손실 계산 (예측 노이즈 vs 실제 노이즈) (MSE)
    noise_loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    return noise_loss

# 손실 함수 레지스트리
loss_registry = {
    "simple": noise_estimation_loss,
}
