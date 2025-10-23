# ================================================================
# functions/denoising.py
# Conditional Diffusion Sampling (BB → T1CE)
# ================================================================
import torch


# ------------------------------------------------------------
# 누적 alpha 계산 함수 (alphā_t)
# ------------------------------------------------------------
def compute_alpha(beta, t):
    """
    Diffusion 과정의 누적 alphā_t 계산.
    (각 step에서의 노이즈 비율 누적값)
    
    Args:
        beta: 전체 β schedule (T,)
        t: 특정 timestep 인덱스 (B,)
    Returns:
        alphā_t: 각 batch에 대한 누적 alpha 값 (B,1,1,1)
    """
    # β 벡터 앞에 0을 붙여서 index alignment 보정
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    
    # α_t = 1 - β_t
    # ᾱ_t = ∏_{i=1}^{t} α_i  (누적곱)
    # -> 각 t마다 누적된 noise-free 비율 (signal 유지량)
    return (1 - beta).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)



# ------------------------------------------------------------
# Conditional Sampling (BB 조건 기반 T1CE 생성)
# ------------------------------------------------------------
def generalized_steps_condition(x, c, e, seq, model, b):
    """
    조건부 Diffusion 샘플링 함수
    (Conditional DDPM — BB를 조건으로 T1CE를 점진적으로 생성)

    Args:
        x: 초기 입력 (noisy image 또는 placeholder)           → (B,1,H,W)
        c: condition image (예: BB MRI slice)                → (B,1,H,W)
        e: 초기 랜덤 노이즈                                   → (B,1,H,W)
        seq: sampling timestep sequence                      → (n_steps,)
        model: 학습된 diffusion UNet 모델
        b: β schedule tensor                                 → (T,)

    Returns:
        xs: 각 timestep별 중간 이미지 리스트
        x0_preds: 각 timestep별 복원된 깨끗한 예측 이미지 리스트
    """

    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])   # t(i), t(i-1) 쌍을 구성하기 위한 next_t 리스트
        x0_preds = []                      # 각 step에서 복원된 x₀ (denoised image) 저장 리스트

        # --------------------------------------------------------
        # Reverse process: noisy → clean 방향으로 반복 (T→0)
        # --------------------------------------------------------
        for idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            # 현재 timestep i, 다음 timestep j 준비
            t = (torch.ones(n) * i).to(x.device)
            t = t.clamp(max=len(b)-1).long()  # 인덱스 범위 안전 보정

            next_t = (torch.ones(n) * j).to(x.device)
            next_t = next_t.clamp(max=len(b)-1).long()

            # ᾱ_t, ᾱ_{t-1} 계산
            at = compute_alpha(b, t)
            at_next = compute_alpha(b, next_t)

            # 첫 step에서만 초기 noisy 샘플 생성
            # (x_t = √ᾱ_t * x₀ + √(1-ᾱ_t) * e)
            if idx == 0:
                xs = [x * at.sqrt() + e * (1.0 - at).sqrt()]

            # 최근 x_t 가져오기 (현재 노이즈 상태)
            xt = xs[-1].to(x.device)
            c = c.to(x.device)  # condition도 같은 device로 이동

            # ----------------------------------------------------
            # 모델 예측: 현재 x_t에서 노이즈 ê_t 추정
            # ----------------------------------------------------
            # 입력 = [noisy image + condition]
            et, _ = model(torch.cat([xt, c], dim=1), t.float())

            # ----------------------------------------------------
            # 복원된 x₀ (denoised prediction) 계산
            # x₀_t = (x_t - √(1-ᾱ_t) * ê_t) / √ᾱ_t
            # ----------------------------------------------------
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))

            # ----------------------------------------------------
            # 다음 step (x_{t-1}) 예측
            # ----------------------------------------------------
            # c1, c2는 DDPM의 sampling variance 조정 term
            c1 = 0 * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()

            # x_{t-1} = √ᾱ_{t-1} * x₀_t + c₁ * noise + c₂ * ê_t
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

            xs.append(xt_next.to("cpu"))  # 다음 step의 noisy 이미지 저장

        # --------------------------------------------------------
        # 모든 step의 noisy 이미지 및 복원된 clean 이미지 반환
        # --------------------------------------------------------
        return xs, x0_preds
