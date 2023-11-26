import datetime
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from drawing_env import DrawingEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor

import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

class TrainCallback(BaseCallback): # https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    """
    각 에피소드가 끝날 때마다 TensorBoard에 이미지를 로깅하는 콜백
    """
    def __init__(self, check_freq, checkpoint_dir):
        super(TrainCallback, self).__init__(verbose=1)
        self.check_freq = check_freq
        self.best_mean_reward = -float('inf')
        self.checkpoint_dir = checkpoint_dir

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    
    def _on_rollout_end(self):
        # rollout이 끝날 때 평균 리워드를 가져와서 TensorBoard에 기록합니다.
        # 이 정보는 `self.model.ep_info_buffer`에서 얻을 수 있습니다.
        # 에피소드 정보가 충분히 쌓였는지 확인합니다.
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            self.logger.record("mean_reward", mean_reward)
            # if mean_reward > self.best_mean_reward:
            #     print("Save best model", mean_reward)
            #     self.best_mean_reward = mean_reward
            #     checkpoint_path = os.path.join(self.checkpoint_dir, f'model_{self.num_timesteps}_steps.zip')
            #     model.save(checkpoint_path)

    
    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            # 환경의 'render' 메서드를 사용하여 현재 이미지를 가져옵니다.
            image_array = self.model.env.envs[0].env.render(mode="rgb_array")
            # # numpy 이미지를 [C, H, W] 포맷의 텐서로 변환합니다.
            image_tensor = np.transpose(image_array, (2, 0, 1))
            # # 이미지를 TensorBoard에 로깅합니다.
            self.tb_formatter.writer.add_image('training/image', image_tensor / 255, self.num_timesteps)
            self.tb_formatter.writer.flush()
        return True
    
    def _on_training_end(self):
        # 콜백이 끝날 때 SummaryWriter를 닫습니다.
        self.tb_formatter.writer.close()

if __name__ == "__main__":
    # Check tensorboard is working with GET http://localhost:6007/
    import requests
    try:
        r = requests.get("http://localhost:6007/")
        print("Tensorboard is running")
    except:
        print("Tensorboard is not running")
        exit(1)

    print("Start training")

    CHECK_FREQ = int(os.getenv("CHECK_FREQ", 100))
    MAX_STEPS= int(os.getenv("MAX_STEPS", 50))
    N_ENV = int(os.getenv("N_ENV", 4))
    TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", 20000) )
    TENSORBOARD_DIR = os.getenv("TENSORBOARD_DIR", "./temp/runs") 
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./temp/ckpt")
    TORCH_HOME = os.getenv("TORCH_HOME", "./temp/cache")
    ALGORITHM = os.getenv("ALGORITHM", "PPO")  # 'PPO' 또는 'SAC'

    print("CHECK_FREQ", CHECK_FREQ)
    print("MAX_STEPS", MAX_STEPS)
    print("N_ENV", N_ENV)
    print("TOTAL_TIMESTEPS", TOTAL_TIMESTEPS)
    print("TENSORBOARD_DIR", TENSORBOARD_DIR)
    print("CHECKPOINT_PATH", CHECKPOINT_PATH)
    print("TORCH_HOME", TORCH_HOME)
    print("ALGORITHM", ALGORITHM)

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    envs = make_vec_env(DrawingEnv, n_envs=N_ENV, seed=0, env_kwargs=dict(max_steps=MAX_STEPS, perceptual_weight=0.5, l2_weight=0.5))
    eval_env = Monitor(DrawingEnv(max_steps=MAX_STEPS))
    print("env num", envs.num_envs)

    dt = datetime.datetime.now().strftime("%m_%d_%H_%M")
    tb_log_name = f"drawing_{dt}"

    if ALGORITHM == "PPO":
        model = PPO("CnnPolicy", envs, verbose=1, tensorboard_log=TENSORBOARD_DIR, device="auto")
    elif ALGORITHM == "SAC":
        model = SAC("CnnPolicy", envs, verbose=1, tensorboard_log=TENSORBOARD_DIR, device="auto", buffer_size=10000)
    else:
        raise ValueError(f"Unsupported algorithm: {ALGORITHM}")
    
    image_callback = TrainCallback(check_freq=CHECK_FREQ, checkpoint_dir=CHECKPOINT_PATH)
    eval_callback = EvalCallback(eval_env, best_model_save_path=CHECKPOINT_PATH, log_path=TENSORBOARD_DIR, eval_freq=CHECK_FREQ)
    callback = CallbackList([eval_callback, image_callback])
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS,  callback=callback, tb_log_name=tb_log_name,  progress_bar=True)
    mean_reward, std_reward = evaluate_policy(model, envs, n_eval_episodes=10)
    model.save("drawing_ppo_model")
    envs.close()