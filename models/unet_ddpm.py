# ================================================================
# models/unet_ddpm.py
# ================================================================
import math  # 수학적 계산을 위한 math 모듈 불러오기
import torch  # PyTorch 텐서 연산을 위한 모듈 불러오기
import torch.nn as nn  # PyTorch의 신경망 구성 모듈 불러오기


# ------------------------------------------------------------
# Timestep embedding (sinusoidal)
# Diffusion 모델에서 시간 스텝을 임베딩하는 함수 정의
# ------------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim):                       # 주어진 시간 스텝을 임베딩 벡터로 변환하는 함수 정의
    assert len(timesteps.shape) == 1                                        # timesteps가 1차원 벡터 형태인지 확인
    half_dim = embedding_dim // 2                                           # 임베딩 차원의 절반 계산
    emb = math.log(10000) / (half_dim - 1)                                  # 주기 스케일 계산
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)     # 주기 벡터 생성
    emb = emb.to(device=timesteps.device)                                   # 입력과 같은 디바이스로 이동
    emb = timesteps.float()[:, None] * emb[None, :]                         # 각 timestep에 주기 스케일 곱하기
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)                # sin과 cos 성분 결합
    if embedding_dim % 2 == 1:                                              # 차원이 홀수인 경우 padding 추가
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb                                                              # 최종 임베딩 반환


# ------------------------------------------------------------
# Common components
# ------------------------------------------------------------
def nonlinearity(x):  # 비선형 활성화 함수 (Swish) 정의
    return x * torch.sigmoid(x)  # Swish: x * sigmoid(x)

def Normalize(in_channels):  # Group Normalization 모듈 반환 함수 정의
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)  # 그룹 정규화 설정


# ------------------------------------------------------------
# Upsample / Downsample blocks
# Upsample = 이미지를 2배로 키우는 모듈
# Downsample = 이미지를 2배로 줄이는 모듈
# Channel 수는 변경하지 않음
# ------------------------------------------------------------
class Upsample(nn.Module):                          # 업샘플링 모듈 정의
    def __init__(self, in_channels, with_conv):     # 입력 채널 수와 Conv 사용 여부 설정
        super().__init__()
        self.with_conv = with_conv                  # conv 사용 여부 저장
        if self.with_conv:                          # conv 사용 시
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)  # 3x3 conv 정의

    def forward(self, x):  # 순전파 정의
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")  # 최근접 보간으로 업샘플
        if self.with_conv:          # conv 사용 시
            x = self.conv(x)        # conv 적용
        return x            # 결과 반환


class Downsample(nn.Module):  # 다운샘플링 모듈 정의
    def __init__(self, in_channels, with_conv):  # 입력 채널 수와 conv 사용 여부 설정
        super().__init__()
        self.with_conv = with_conv                                      # conv 사용 여부 저장
        if self.with_conv:                                              # conv 사용 시
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)    # stride=2 conv 정의

    def forward(self, x):  # 순전파 정의
        if self.with_conv:  # conv 사용 시
            x = nn.functional.pad(x, (0, 1, 0, 1))  # 크기 맞춤을 위한 패딩 추가
            x = self.conv(x)  # conv 적용
        else:  # conv 미사용 시
            x = nn.functional.avg_pool2d(x, 2, 2)  # 평균 풀링으로 다운샘플링
        return x  # 결과 반환


# ------------------------------------------------------------
# Residual + Attention blocks
# 입력 채널, 출력 채널, dropout 비율, timestep embedding 채널 수 지정
# ------------------------------------------------------------
class ResnetBlock(nn.Module):   # ResNet 기반 블록 정의
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()                                                              # 부모 클래스 초기화
        self.in_channels = in_channels                                                  # 입력 채널 수 저장
        out_channels = in_channels if out_channels is None else out_channels            # 출력 채널 결정
        self.out_channels = out_channels                                                # 출력 채널 저장
        self.use_conv_shortcut = conv_shortcut                                          # shortcut 경로에서 conv 사용 여부

        self.norm1 = Normalize(in_channels)                             # 첫 번째 정규화
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)      # 첫 번째 conv
        self.temb_proj = nn.Linear(temb_channels, out_channels)         # timestep 임베딩 투영 레이어
        self.norm2 = Normalize(out_channels)                            # 두 번째 정규화
        self.dropout = nn.Dropout(dropout)                              # 드롭아웃 적용
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)     # 두 번째 conv

        if self.in_channels != self.out_channels:           # 입출력 채널 다를 경우
            if self.use_conv_shortcut:                      # conv shortcut 사용 시
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)  # 3x3 conv 정의
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)  # 1x1 conv 정의

    def forward(self, x, temb):  # 순전파 정의
        h = self.norm1(x)  # 입력 정규화
        h = nonlinearity(h)  # swish 활성화
        h = self.conv1(h)  # 첫 번째 conv 수행
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]  # 시간 임베딩 추가
        h = self.norm2(h)  # 정규화
        h = nonlinearity(h)  # swish 활성화
        h = self.dropout(h)  # dropout 적용
        h = self.conv2(h)  # 두 번째 conv 수행

        if self.in_channels != self.out_channels:  # 차원 다를 경우 skip 경로 맞추기
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h  # skip connection으로 출력 결합


class AttnBlock(nn.Module):                                     # self-attention 블록 정의
    def __init__(self, in_channels):                            # 입력 채널 수 설정
        super().__init__()  
        self.in_channels = in_channels                          # 채널 수 저장
        self.norm = Normalize(in_channels)                      # 정규화 레이어
        self.q = nn.Conv2d(in_channels, in_channels, 1)         # Query 생성 conv
        self.k = nn.Conv2d(in_channels, in_channels, 1)         # Key 생성 conv
        self.v = nn.Conv2d(in_channels, in_channels, 1)         # Value 생성 conv
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)  # 출력 투영 conv

    def forward(self, x):  # 순전파 정의
        h_ = self.norm(x)  # 입력 정규화
        q, k, v = self.q(h_), self.k(h_), self.v(h_)  # Q, K, V 계산
        b, c, h, w = q.shape  # 배치, 채널, 높이, 너비 추출

        q = q.reshape(b, c, h*w).permute(0, 2, 1)  # Q를 (B, HW, C)로 변형
        k = k.reshape(b, c, h*w)  # K를 (B, C, HW)로 변형
        w_ = torch.bmm(q, k) * (c ** -0.5)  # scaled dot-product 계산
        w_ = nn.functional.softmax(w_, dim=2)  # attention 확률화

        v = v.reshape(b, c, h*w)  # V를 (B, C, HW)로 변형
        w_ = w_.permute(0, 2, 1)  # 가중치 전치
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)  # attention 적용 후 (B, C, H, W) 복원
        return x + self.proj_out(h_)  # residual connection으로 출력 결합


# ------------------------------------------------------------
# UNet Backbone
# ------------------------------------------------------------
class Model(nn.Module):  # 전체 UNet 모델 정의
    def __init__(self, config):  # 설정값 입력
        super().__init__()  # 부모 클래스 초기화
        self.config = config  # 설정 저장
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)  # 채널 설정
        num_res_blocks = config.model.num_res_blocks  # ResNet 블록 개수
        attn_resolutions = config.model.attn_resolutions  # attention이 적용될 해상도
        dropout = config.model.dropout  # 드롭아웃 비율
        in_channels = config.model.in_channels  # 입력 채널 수
        resolution = config.data.image_size  # 입력 이미지 크기
        resamp_with_conv = config.model.resamp_with_conv  # 업/다운샘플링 시 conv 사용 여부

        self.ch = ch  # 기본 채널 수 저장
        self.temb_ch = self.ch * 4  # timestep embedding 채널 수 설정
        self.num_resolutions = len(ch_mult)  # 해상도 단계 수
        self.num_res_blocks = num_res_blocks  # 각 해상도당 블록 수
        self.resolution = resolution  # 입력 해상도 저장
        self.in_channels = in_channels  # 입력 채널 저장

        # timestep embedding MLP 정의
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),  # 1차 선형층
            nn.Linear(self.temb_ch, self.temb_ch),  # 2차 선형층
        ])

        # ======================================================
        # Downsampling Path (Encoder)
        # ======================================================
        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, 1, 1)  # 첫 conv 레이어

        curr_res = resolution  # 현재 해상도 설정
        in_ch_mult = (1,) + ch_mult  # 입력 채널 배수 설정
        self.down = nn.ModuleList()  # 다운샘플 블록 리스트 초기화

        block_in = None
        for i_level in range(self.num_resolutions):                                         # 각 해상도 단계에 대해 반복
            block = nn.ModuleList()                                                         # 블록 리스트
            attn = nn.ModuleList()                                                          # attention 리스트
            block_in = ch * in_ch_mult[i_level]                                             # 입력 채널 계산
            block_out = ch * ch_mult[i_level]                                               # 출력 채널 계산
            for i_block in range(self.num_res_blocks):                                      # 블록 반복
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))      # ResnetBlock 추가
                block_in = block_out                                                        # 입력 채널 갱신
                if curr_res in attn_resolutions:                                            # 해당 해상도에 attention 적용 시
                    attn.append(AttnBlock(block_in))                                        # attention 블록 추가
            down = nn.Module()
            down.block = block                                                      # 블록 연결
            down.attn = attn                                                        # attention 연결
            if i_level != self.num_resolutions - 1:                                 # 마지막 단계가 아니면
                down.downsample = Downsample(block_in, resamp_with_conv)                # 다운샘플 추가
                curr_res //= 2                                                          # 해상도 절반 감소
            self.down.append(down)                                                  # 전체 down 리스트에 추가

        # ======================================================
        # Middle (Bottleneck)
        # ======================================================
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, dropout=dropout, temb_channels=self.temb_ch)  # 첫 feature refine & timestep 정보 융합
        self.mid.attn_1 = AttnBlock(block_in)  #  Global Dependency 학습
        self.mid.block_2 = ResnetBlock(in_channels=block_in, dropout=dropout, temb_channels=self.temb_ch)  # Global Dependency feature 세밀하게 다듬기

        # ======================================================
        # Upsampling Path (Decoder)
        # ======================================================
        self.up = nn.ModuleList()  # 업샘플 블록 리스트
        for i_level in reversed(range(self.num_resolutions)):  # 해상도 단계 역순 반복
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):  # skip 연결 포함 추가 블록
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]  # skip 입력 채널 계산
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))  # 업샘플 블록 추가
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))  # attention 추가
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)  # 업샘플 블록 추가
                curr_res *= 2  # 해상도 2배 증가
            self.up.insert(0, up)  # 앞쪽에 삽입하여 decoder 순서 유지

        # ======================================================
        # Output Head
        # ======================================================
        self.norm_out = Normalize(block_in)  # 마지막 정규화
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)  # 출력 conv

    def forward(self, x, t):  # 순전파 함수 정의
        assert x.shape[2] == x.shape[3] == self.resolution  # 입력 해상도 검증

        temb = get_timestep_embedding(t, self.ch)  # timestep 임베딩 계산
        temb = self.temb.dense[0](temb)  # 첫 linear 적용
        temb = nonlinearity(temb)  # swish 활성화
        temb = self.temb.dense[1](temb)  # 두 번째 linear 적용

        hs = [self.conv_in(x)]  # 첫 conv 후 특징 저장
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)  # down path ResBlock 적용
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)  # attention 적용
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))  # 다운샘플 적용

        h = hs[-1]  # bottleneck 입력
        h_mid_1 = self.mid.block_1(h, temb)  # 중간 ResBlock1
        h = self.mid.attn_1(h_mid_1)  # attention 적용
        h_mid_2 = self.mid.block_2(h, temb)  # 중간 ResBlock2

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)  # skip 연결 및 블록 처리
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)  # attention 적용
            if i_level != 0:
                h = self.up[i_level].upsample(h)  # 업샘플 적용

        h = self.norm_out(h)  # 출력 정규화
        h = nonlinearity(h)  # 활성화
        h = self.conv_out(h)  # 최종 conv 출력

        return h, (h_mid_1, h_mid_2)  # 출력 및 중간 feature 반환
