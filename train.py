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
import gymnasium as gym


load_dotenv()


class CustomCallback(BaseCallback): # https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    """
    각 에피소드가 끝날 때마다 TensorBoard에 이미지를 로깅하는 콜백
    """
    def __init__(self, check_freq, difficulty_inc_period, checkpoint_dir):
        super(CustomCallback, self).__init__(verbose=1)
        self.check_freq = check_freq
        self.best_mean_reward = -float('inf')
        self.checkpoint_dir = checkpoint_dir
        self.difficulty_inc_period = difficulty_inc_period
        self.actions = []
        

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
    
    
    def _on_step(self):
        # 모든 스텝에서 액션을 계산합니다.
        current_action = self.model.env.actions.mean(axis=0)

        # 첫 번째 스텝에서 actions_dim을 초기화합니다.
        if not hasattr(self, 'actions_dim'):
            self.actions_dim = [[] for _ in range(current_action.shape[0])]

        # 각 차원별로 액션 값을 추가합니다.
        for dim in range(current_action.shape[0]):
            self.actions_dim[dim].append(current_action[dim])

        if self.num_timesteps % self.check_freq == 0:
            # 환경의 'render' 메서드를 사용하여 현재 이미지를 가져옵니다.
            image_array = self.model.env.envs[0].env.render(mode="rgb_array")
            # numpy 이미지를 [C, H, W] 포맷의 텐서로 변환합니다.
            image_tensor = np.transpose(image_array, (2, 0, 1))
            # 이미지를 TensorBoard에 로깅합니다.
            self.tb_formatter.writer.add_image('sample/image', image_tensor / 255, self.num_timesteps)
            self.tb_formatter.writer.flush()
            
            # 각 차원별로 히스토그램을 로깅합니다.
            for dim, actions in enumerate(self.actions_dim):
                if actions:
                    if dim < 8: # cooorid
                        # self.tb_formatter.writer.add_histogram(f'actions/histogram_coord_{dim}', np.array(actions), self.num_timesteps)
                        pass
                    elif dim == 8:
                        self.tb_formatter.writer.add_histogram(f'actions/thickness_s', np.array(actions), self.num_timesteps)
                    elif dim == 9:
                        self.tb_formatter.writer.add_histogram(f'actions/thickness_e', np.array(actions), self.num_timesteps)
                    elif dim == 10:
                        self.tb_formatter.writer.add_histogram(f'actions/r', np.array(actions), self.num_timesteps)
                    elif dim == 11:
                        self.tb_formatter.writer.add_histogram(f'actions/g', np.array(actions), self.num_timesteps)
                    elif dim == 12:
                        self.tb_formatter.writer.add_histogram(f'actions/b', np.array(actions), self.num_timesteps)
                    elif dim == 13:
                        self.tb_formatter.writer.add_histogram(f'actions/a', np.array(actions), self.num_timesteps)
                    elif dim == 14:
                        self.tb_formatter.writer.add_histogram(f'actions/stroke_length', np.array(actions), self.num_timesteps)
                    self.actions_dim[dim].clear()

        if self.num_timesteps % self.difficulty_inc_period == 0:
            DrawingEnv.inc_difficulty()

        return True
    
    def _on_training_end(self):
        # 콜백이 끝날 때 SummaryWriter를 닫습니다.
        self.tb_formatter.writer.close()

if __name__ == "__main__":
    # Check tensorboard is working with GET http://localhost:6007/
    # import requests
    # try:
    #     r = requests.get("http://localhost:6007/")
    #     print("Tensorboard is running")
    # except:
    #     print("Tensorboard is not running")
    #     exit(1)

    print("Start training")

    CHECK_FREQ = int(os.getenv("CHECK_FREQ", 100))
    MAX_STEPS= int(os.getenv("MAX_STEPS", 50))
    N_ENV = int(os.getenv("N_ENV", 4))
    TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", 20000) )
    TENSORBOARD_DIR = os.getenv("TENSORBOARD_DIR", "./temp/runs") 
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./temp/ckpt")
    TORCH_HOME = os.getenv("TORCH_HOME", "./temp/cache")
    ALGORITHM = os.getenv("ALGORITHM", "PPO")  # 'PPO' 또는 'SAC'
    SAC_BUFFER_SIZE = int(os.getenv("SAC_BUFFER_SIZE", 10000))  # 'PPO' 또는 'SAC'
    DIFFICULTY_INC_PERIOD = int(os.getenv("DIFFICULTY_INC_PERIOD", 10000))

    print("CHECK_FREQ", CHECK_FREQ)
    print("DIFFICULTY_INC_PERIOD", DIFFICULTY_INC_PERIOD)
    print("MAX_STEPS", MAX_STEPS)
    print("N_ENV", N_ENV)
    print("TOTAL_TIMESTEPS", TOTAL_TIMESTEPS)
    print("TENSORBOARD_DIR", TENSORBOARD_DIR)
    print("CHECKPOINT_PATH", CHECKPOINT_PATH)
    print("TORCH_HOME", TORCH_HOME)
    print("ALGORITHM", ALGORITHM)
    print("SAC_BUFFER_SIZE", SAC_BUFFER_SIZE)

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    envs = make_vec_env(DrawingEnv, n_envs=N_ENV, seed=0, 
                        env_kwargs=dict(max_steps=MAX_STEPS, perceptual_weight=0.9, l2_weight=0.1))
    eval_env = Monitor(DrawingEnv(max_steps=MAX_STEPS))
    print("env num", envs.num_envs)

    dt = datetime.datetime.now().strftime("%m_%d_%H_%M")
    tb_log_name = f"drawing_{dt}"

    if ALGORITHM == "PPO":
        model = PPO("CnnPolicy", envs, verbose=0, tensorboard_log=TENSORBOARD_DIR, device="auto")
    elif ALGORITHM == "SAC":
        model = SAC("CnnPolicy", envs, verbose=0, tensorboard_log=TENSORBOARD_DIR, device="auto", buffer_size=SAC_BUFFER_SIZE)
    else:
        raise ValueError(f"Unsupported algorithm: {ALGORITHM}")
    
    custom_callback = CustomCallback(check_freq=CHECK_FREQ, difficulty_inc_period=DIFFICULTY_INC_PERIOD, checkpoint_dir=CHECKPOINT_PATH)
    eval_callback = EvalCallback(eval_env, best_model_save_path=CHECKPOINT_PATH, log_path=TENSORBOARD_DIR, eval_freq=CHECK_FREQ)
    callback = CallbackList([eval_callback, custom_callback])
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS,  callback=callback, tb_log_name=tb_log_name,  progress_bar=True)
    mean_reward, std_reward = evaluate_policy(model, envs, n_eval_episodes=10)
    model.save("drawing_ppo_model")
    envs.close()