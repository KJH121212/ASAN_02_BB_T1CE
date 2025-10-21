# ================================================================
# functions/ckpt_util.py
# 체크포인트 / 다운로드 관련 함수
# ================================================================
import os
import hashlib
import requests
from tqdm import tqdm

# 공개 모델 URL 및 해시 목록
URL_MAP = {
    "cifar10": "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1",
    "ema_cifar10": "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1",
    "lsun_church": "https://heibox.uni-heidelberg.de/f/2711a6f712e34b06b9d8/?dl=1",
    "ema_lsun_church": "https://heibox.uni-heidelberg.de/f/44ccb50ef3c6436db52e/?dl=1",
}
CKPT_MAP = {
    "cifar10": "diffusion_cifar10_model/model-790000.ckpt",
    "ema_cifar10": "ema_diffusion_cifar10_model/model-790000.ckpt",
    "lsun_church": "diffusion_lsun_church_model/model-4432000.ckpt",
    "ema_lsun_church": "ema_diffusion_lsun_church_model/model-4432000.ckpt",
}
MD5_MAP = {
    "cifar10": "82ed3067fd1002f5cf4c339fb80c4669",
    "ema_cifar10": "1fa350b952534ae442b1d5235cce5cd3",
    "lsun_church": "eb619b8a5ab95ef80f94ce8a5488dae3",
    "ema_lsun_church": "fdc68a23938c2397caba4a260bc2445f",
}


def download(url, local_path, chunk_size=1024):
    """모델 가중치 다운로드"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))


def md5_hash(path):
    """MD5 해시 계산"""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_ckpt_path(name, root=None, check=False):
    """모델 이름으로 체크포인트 다운로드 및 경로 반환"""
    if 'church_outdoor' in name:
        name = name.replace('church_outdoor', 'church')
    assert name in URL_MAP
    cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    root = root or os.path.join(cachedir, "diffusion_models_converted")
    path = os.path.join(root, CKPT_MAP[name])

    if not os.path.exists(path) or (check and md5_hash(path) != MD5_MAP[name]):
        print(f"Downloading {name} from {URL_MAP[name]}")
        download(URL_MAP[name], path)
        assert md5_hash(path) == MD5_MAP[name]
    return path
