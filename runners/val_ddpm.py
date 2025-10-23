import os
import torch
import logging
from torch.utils.data import DataLoader
from datasets.val_dataset import Val_Dataset        # 방금 만든 validation dataset
from models.diffusion import Model
from models.ema import EMAHelper
from functions.losses import loss_registry
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np


def val_diffusion(config, device):
    """
    Validation routine for Diffusion model.
    입력: (BB_clean, BB_noisy)
    정답: T1CE
    """

    # ------------------------------------------------------------
    # 1️⃣ 데이터 로드
    # ------------------------------------------------------------
    data_dir = config.data.model_dir_path
    bb_dir = os.path.join(data_dir, config.data.bb_dir)
    t1ce_dir = os.path.join(data_dir, config.data.t1ce_dir)

    val_dataset = Val_Dataset(bb_dir=bb_dir, t1ce_dir=t1ce_dir, img_size=config.data.image_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # ------------------------------------------------------------
    # 2️⃣ 모델 및 EMA 초기화
    # ------------------------------------------------------------
    model = Model(config).to(device)
    model = torch.nn.DataParallel(model)

    # EMA 모델 불러오기
    ema_helper = EMAHelper(mu=config.model.ema_rate) if config.model.ema else None
    if ema_helper:
        ema_helper.register(model)

    # Checkpoint 로드
    ckpt_path = os.path.join(config.model.log_path, "ckpt.pth")
    assert os.path.exists(ckpt_path), f"❌ checkpoint 없음: {ckpt_path}"

    states = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(states[0])
    if ema_helper and len(states) > 4:
        ema_helper.load_state_dict(states[4])
        ema_helper.ema(model)

    model.eval()

    # ------------------------------------------------------------
    # 3️⃣ Diffusion schedule 준비
    # ------------------------------------------------------------
    betas = torch.from_numpy(config.diffusion.betas).float().to(device)
    num_timesteps = betas.shape[0]

    # ------------------------------------------------------------
    # 4️⃣ Validation 루프
    # ------------------------------------------------------------
    total_loss, total_psnr, total_ssim = [], [], []

    with torch.no_grad():  # Validation 단계에서는 학습(gradient update)이 없으므로 gradient 계산을 비활성화
        for batch_idx, (x, y, meta) in enumerate(val_loader):  # DataLoader에서 한 배치씩 반복 (x: 입력, y: 정답, meta: 메타데이터)
            
            bb_clean = x[:, 0:1].to(device)   # 첫 번째 채널: 깨끗한 BB (Clean version)
            bb_noisy = x[:, 1:2].to(device)   # 두 번째 채널: 노이즈가 추가된 BB (Noisy version)
            target = y.to(device)             # T1CE Ground Truth (정답)

            # ------------------------------------------------------------
            # 2️⃣ Diffusion step 준비
            # ------------------------------------------------------------
            n = bb_noisy.size(0)                                       # 현재 batch 크기 (샘플 수)
            t = torch.randint(low=0, high=num_timesteps, size=(n,))    # 무작위 timestep t 선택 (0~T-1 사이)
            t = t.to(device)                                           # GPU로 이동
            e = torch.randn_like(bb_noisy).float()                     # 동일 shape의 Gaussian noise 생성
            b = betas                                                  # beta 스케줄 불러오기

            # ------------------------------------------------------------
            # 3️⃣ 모델 예측 (노이즈 예측 단계)
            # ------------------------------------------------------------
            # 입력: [BB_noisy, BB_clean] 두 채널을 concat → 모델은 noisy image로부터 노이즈를 예측
            pred_noise, _ = model(torch.cat([bb_noisy, bb_clean], dim=1), t.float())

            # ------------------------------------------------------------
            # 4️⃣ 노이즈 제거 및 복원
            # ------------------------------------------------------------
            # α_t 계산: 각 timestep별 누적 (1 - β_t)의 곱 → a_t = ∏(1 - β)
            a_t = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

            # 역과정 복원: x₀ = (x_t - √(1 - α_t) * ε_pred) / √(α_t)
            # 즉, noisy image에서 예측된 노이즈(pred_noise)를 제거하여 복원 이미지(recon) 생성
            recon = (bb_noisy - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()

            # ------------------------------------------------------------
            # 5️⃣ 손실 계산 (MSE)
            # ------------------------------------------------------------
            # 예측 복원 이미지(recon)와 실제 T1CE(target) 간의 Mean Squared Error 계산
            loss = (recon - target).pow(2).mean()
            total_loss.append(loss.item())  # 평균 손실값을 리스트에 저장 (추후 평균 산출용)

            # ------------------------------------------------------------
            # 6️⃣ 품질 평가 (PSNR / SSIM)
            # ------------------------------------------------------------
            recon_np = recon.squeeze().cpu().numpy()   # GPU 텐서를 CPU로 옮기고 numpy 배열로 변환
            target_np = target.squeeze().cpu().numpy() # 동일하게 변환

            # PSNR (Peak Signal-to-Noise Ratio): 복원 영상의 노이즈 대비 품질 평가
            total_psnr.append(psnr(target_np, recon_np, data_range=2.0))

            # SSIM (Structural Similarity Index): 구조적 유사도 평가
            total_ssim.append(ssim(target_np, recon_np, data_range=2.0))

            # ------------------------------------------------------------
            # 7️⃣ 로그 출력
            # ------------------------------------------------------------
            logging.info(
                f"[VAL] Case {batch_idx+1}/{len(val_loader)} "     # 현재 케이스 인덱스 출력
                f"Loss={loss.item():.6f}, "                        # 손실 값
                f"PSNR={total_psnr[-1]:.2f}, "                     # 최근 계산된 PSNR
                f"SSIM={total_ssim[-1]:.3f}"                       # 최근 계산된 SSIM
            )

    # ------------------------------------------------------------
    # 5️⃣ 평균 성능 출력
    # ------------------------------------------------------------
    avg_loss = np.mean(total_loss)
    avg_psnr = np.mean(total_psnr)
    avg_ssim = np.mean(total_ssim)

    logging.info(f"✅ Validation 완료 — 평균 손실: {avg_loss:.6f}, 평균 PSNR: {avg_psnr:.2f}, 평균 SSIM: {avg_ssim:.3f}")
    print(f"✅ Validation 완료 — 평균 손실: {avg_loss:.6f}, 평균 PSNR: {avg_psnr:.2f}, 평균 SSIM: {avg_ssim:.3f}")

    return avg_loss, avg_psnr, avg_ssim
