FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# =========================================================
# 환경 변수 설정
# =========================================================
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =========================================================
# Kakao PyPI 미러 + pip 최신화 + jupyter 설치
# =========================================================
RUN printf "%s\n" \
    "[global]" \
    "index-url=https://mirror.kakao.com/pypi/simple/" \
    "extra-index-url=https://pypi.org/simple/" \
    "trusted-host=mirror.kakao.com" \
    > /etc/pip.conf \
    && pip install --no-cache-dir -U pip setuptools wheel jupyter

# =========================================================
# 필수 시스템 패키지 (빌드 툴 포함)
# =========================================================
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update -qq && apt install -qqy\
        sudo\
        tzdata\
        vim\
        curl\
        jq\
        git\
        libgl1-mesa-glx\
        libglib2.0-0\
    && apt clean && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

# =========================================================
# Python 패키지 설치
# =========================================================
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    numpy \
    pandas \
    tqdm \
    matplotlib \
    wandb \
    SimpleITK \
    omegaconf \
    scipy \
    pytorch-fid