import torch.optim as optim  # PyTorch의 최적화(optimizer) 모듈 불러오기

def get_optimizer(config, parameters):          # 설정(config)과 모델 파라미터를 받아 optimizer를 생성하는 함수
    if config.optim.optimizer == 'Adam':            # Optimizer가 'Adam'일 경우
        return optim.Adam(                              # Adam optimizer 생성 및 반환
            parameters,                                     # 모델 파라미터 전달
            lr=config.optim.lr,                             # 학습률(learning rate)
            weight_decay=config.optim.weight_decay,         # L2 정규화 항 (가중치 감쇠)
            betas=(config.optim.beta1, 0.999),              # 1차, 2차 모멘텀 계수
            amsgrad=config.optim.amsgrad,                   # AMSGrad 사용 여부
            eps=config.optim.eps                            # 수치 안정성용 epsilon 값
        )
    elif config.optim.optimizer == 'RMSProp':   # Optimizer가 'RMSProp'일 경우
        return optim.RMSprop(                       # RMSProp optimizer 생성
            parameters,                                 # 모델 파라미터
            lr=config.optim.lr,                         # 학습률
            weight_decay=config.optim.weight_decay      # L2 정규화 항
        )
    elif config.optim.optimizer == 'SGD':       # Optimizer가 'SGD'일 경우
        return optim.SGD(                           # 확률적 경사하강법 optimizer 생성
            parameters,                                 # 모델 파라미터
            lr=config.optim.lr,                         # 학습률
            momentum=0.9                                # 모멘텀 값(기본적으로 0.9로 설정)
        )
    else:  # 위 세 가지 외의 Optimizer가 설정된 경우
        raise NotImplementedError(  # 구현되지 않은 Optimizer에 대해 에러 발생
            f'Optimizer {config.optim.optimizer} not understood.'
        )
