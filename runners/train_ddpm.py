import os
import time
import logging
import torch
from torch.utils.data import DataLoader
from datasets.concatbbt1ce import ConcatBBT1CEDataset
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry


def train_diffusion(config, device):
    # 데이터 경로 설정
    data_dir = config.data.model_dir_path                               # 데이터 루트 폴더
    bb_dir = os.path.join(data_dir, config.data.bb_dir)                 # BB 이미지 폴더
    t1ce_dir = os.path.join(data_dir, config.data.t1ce_dir)             # T1CE 이미지 폴더

    # 데이터셋 및 DataLoader 정의
    dataset = ConcatBBT1CEDataset(bb_dir=bb_dir, t1ce_dir=t1ce_dir)     # (BB, T1CE) 쌍 데이터셋 생성
    train_loader = DataLoader(
        dataset,
        batch_size=1,                                                  # 한 번에 하나의 환자(volume) 로드
        shuffle=True,                                                  # 학습 시 순서 랜덤
        num_workers=config.data.num_workers,                           # 병렬 데이터 로딩
    )

    # 모델 및 Optimizer 초기화
    model = Model(config).to(device)                                   # Diffusion 모델 로드 후 GPU로 이동
    model = torch.nn.DataParallel(model)                               # 멀티 GPU 지원 (DataParallel)
    optimizer = get_optimizer(config, model.parameters())              # 설정파일 기반 optimizer 생성

    # EMA (Exponential Moving Average) 설정
    ema_helper = EMAHelper(mu=config.model.ema_rate) if config.model.ema else None
    if ema_helper:
        ema_helper.register(model)                                     # 모델 파라미터 EMA 등록
    else:
        ema_helper = None

    # 학습 재시작(resume) 기능
    start_epoch, step = 0, 0                                           # 기본값 초기화
    if config.model.resume_training:                                   # 이어서 학습할 경우
        ckpt_path = os.path.join(config.model.log_path)                # 체크포인트 경로 설정
        states = torch.load(ckpt_path, map_location=device)            # 체크포인트 로드
        model.load_state_dict(states[0])                               # 모델 가중치 복원
        optimizer.load_state_dict(states[1])                           # 옵티마이저 상태 복원
        start_epoch = states[2]                                        # epoch 위치 복원
        step = states[3]                                               # step 위치 복원
        if ema_helper:
            ema_helper.load_state_dict(states[4])                      # EMA 파라미터 복원

    # ------------------------------------------------------------
    # ✅ 6️⃣ Diffusion 스케줄 설정
    # ------------------------------------------------------------
    betas = torch.from_numpy(config.diffusion.betas).float().to(device) # beta 스케줄을 텐서로 변환
    num_timesteps = betas.shape[0]                                      # 전체 타임스텝 개수 저장
    tb_logger = config.tb_logger                                        # TensorBoard 로거 가져오기
    log_path = config.model.log_path                                    # 로그 및 모델 저장 경로

    # ------------------------------------------------------------
    # ✅ 7️⃣ 학습 루프 시작
    # ------------------------------------------------------------
    for epoch in range(start_epoch, config.training.n_epochs):          # 전체 epoch 반복
        data_start = time.time()                                        # 데이터 로딩 시간 측정 시작
        data_time = 0

        for i, images in enumerate(train_loader):                       # 각 배치(batch) 단위 반복
            # --------------------------------------------------------
            # [B, C=2, Z, H, W]
            # C=0 → BB,  C=1 → T1CE (target)
            # --------------------------------------------------------
            src = images[:, 0:1, :, :, :]                              # BB (입력)
            tar = images[:, 1:2, :, :, :]                              # T1CE (정답)

            # Z축 단위로 일정한 슬라이스 크기(batch_size)씩 잘라 처리
            for z_axis in range(0, src.shape[2], config.training.batch_size):
                # (Z 방향으로 batch_size 만큼 자르기)
                src_2d = src[:, :, z_axis:z_axis + config.training.batch_size, :, :]
                tar_2d = tar[:, :, z_axis:z_axis + config.training.batch_size, :, :]

                # (B, 1, Zb, H, W) → (B*Zb, 1, H, W) 형태로 변환
                B, C, Zb, H, W = src_2d.shape
                src_2d = src_2d.permute(0, 2, 1, 3, 4).reshape(B * Zb, C, H, W)
                tar_2d = tar_2d.permute(0, 2, 1, 3, 4).reshape(B * Zb, C, H, W)

                # GPU로 이동
                x_c = src_2d.float().to(device)                         # 조건부 입력 (BB)
                x = tar_2d.float().to(device)                           # 목표 출력 (T1CE)

                n = x.size(0)                                           # 현재 batch 내 샘플 수
                data_time += time.time() - data_start                   # 데이터 준비 시간 누적
                model.train()                                           # 학습 모드로 전환
                step += 1                                               # global step 증가

                # 노이즈 및 timestep 샘플링
                e = torch.randn_like(x).float()                         # 가우시안 노이즈 생성
                b = betas                                               # 베타 스케줄 사용

                # timestep t 무작위 샘플링 (antithetic 방식)
                t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]    # 대칭적 timestep 구성

                # 손실(loss) 계산
                loss = loss_registry[config.model.type](model, x, x_c, t, e, b)
                tb_logger.add_scalar("loss", loss, global_step=step)    # TensorBoard 기록

                # 로깅 및 출력
                logging.info(
                    f"step: {step}, loss: {loss.item():.6f}, "
                    f"data time: {data_time / (i + 1):.3f}, "
                    f"t: {t[0].item()}, epoch: {epoch}"
                )

                # 옵티마이저 업데이트
                optimizer.zero_grad()                                   # 기존 gradient 초기화
                loss.backward()                                         # 역전파 (gradient 계산)

                torch.nn.utils.clip_grad_norm_(                        # gradient clipping
                    model.parameters(), config.optim.grad_clip
                )

                optimizer.step()                                        # 파라미터 업데이트

                # EMA 적용 (모델 파라미터 지수이동평균 업데이트)
                if ema_helper:
                    ema_helper.update(model)

                data_start = time.time()                                # 다음 루프용 타이머 초기화

        # Checkpoint 저장
        states = [                                                     # 저장할 상태 목록
            model.state_dict(),                                        # 모델 가중치
            optimizer.state_dict(),                                    # 옵티마이저 상태
            epoch,                                                     # 현재 epoch
            step,                                                      # 전체 step 수
        ]
        if ema_helper:
            states.append(ema_helper.state_dict())                     # EMA 상태 추가

        os.makedirs(log_path, exist_ok=True)                           # 저장 폴더 생성
        torch.save(states, os.path.join(log_path, f"ckpt_{epoch}.pth"))# epoch별 저장
        torch.save(states, os.path.join(log_path, "ckpt.pth"))         # 최신 모델 덮어쓰기

        logging.info(f"Epoch {epoch} 완료 ✅ — checkpoint 저장됨: {log_path}")