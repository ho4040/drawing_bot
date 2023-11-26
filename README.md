
### drawing_env.py

* BezierDrawing : 획을 그려주는 기능을 구현
* DrawingEnv : gymnasium 인터페이스로 Environment 구현

### train.py

* stable_baselines3 를 이용한 학습


### 환경변수

* CHECK_FREQ : 텐서보드기록 주기 (기본값: 5000) 
* MAX_STEPS : 에피소드 길이 (기본값: 50) 
* N_ENV : 환경 동시에 몇 개 돌릴지 (기본값: 4)
* TOTAL_TIMESTEPS : 총 학습 스텝 (기본값: 200000)
* TENSORBOARD_DIR : 텐서보드 기록 경로 (기본값: ./temp/runs)
* TORCH_HOME : vgg weight 다운로드 경루 (기본값: ./temp/cache)
* CHECKPOINT_PATH : 체크포인트 저장경로 (기본값: ./temp/ckpt)

### 포트

* 텐서보드: 6007

### 실행

`stable_baselines.common.env_checker` 를 이용하여 env 검사

```bash
python drawing_env.py
```


선이 잘 그려지는지 확인
```bash
python test_drawing.py
```

그림이 잘 그려지는지 확인

```bash
python test_env.py
```

리워드와 Loss 계산이 잘 되는지  확인

```bash
python test_loss.py
```

학습 (CPU)

```bash
tensorboard --logdir=./temp/runs --port 6007
```
텐서보드를 먼저 켜야함.

```bash
python train.py
```

도커 빌드
```bash
./docker_build.sh
```

Docker + GPU 학습.
```bash
./docker_run.sh
```

