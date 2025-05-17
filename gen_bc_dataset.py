import argparse
import os
import json
import numpy as np
from drawing_env import DrawingEnv, BezierDrawingCanvas
from tqdm import tqdm
from PIL import Image
import shutil

def ensure_dir(directory):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def remove_dir(directory):
    """디렉토리가 있으면 제거"""
    if os.path.exists(directory):
        shutil.rmtree(directory)

def save_action(action, filepath):
    """액션 데이터를 JSON 형식으로 저장"""
    action_dict = {
        'control_points': action[:8].reshape(4, 2).tolist(),  # 4개의 컨트롤 포인트
        'start_width': float(action[8]),
        'end_width': float(action[9]),
        'color': action[10:14].tolist(),  # RGBA
        'num_points': int(action[14] * 100) + 10  # 곡선의 점 개수
    }
    with open(filepath, 'w') as f:
        json.dump(action_dict, f, indent=2)

def generate_sample(env, num_previous_strokes, num_after_strokes, gen_true_img=False):
    """단일 샘플 생성
    Args:
        env: DrawingEnv 인스턴스
        num_previous_strokes: 예측해야 할 액션 전에 그려야 할 스트로크의 수
        num_after_strokes: 액션 이후에 추가로 그려야 할 스트로크의 수. 타겟 이미지를 더 어렵게 만들기 위함
        gen_true_img: true_img 생성 여부
    Returns:
        sample: 현재 이미지, 목표 이미지, 액션을 포함하는 딕셔너리
    """
    # 초기 상태 설정
    env.reset()
    env.canvas.clear_surface()
    sample = {}
    
    # 스트로크 그리기
    total_strokes = num_previous_strokes + num_after_strokes + 1
    for i in range(total_strokes):
        action = env.random_action()
        
        if i == num_previous_strokes:
            observ = env.get_observation()  # [3, 224, 224*2] ndarray
            sample["curr_img"] = observ[:, :, 224:].copy()  # CHW format [3, 224, 224]
            sample["action"] = action.copy()
        
        env.step(action) # 액션 적용
        
        # true_img 생성 (정답 이미지)
        if gen_true_img and i == num_previous_strokes: 
            observ = env.get_observation()  # [3, 224, 224*2] ndarray
            sample["true_img"] = observ[:, :, 224:].copy()  # CHW format [3, 224, 224]
            
        # 마지막 스트로크 후의 상태를 목표 이미지로 저장
        if i == total_strokes - 1:
            observ = env.get_observation()  # [3, 224, 224*2] ndarray
            sample["goal_img"] = observ[:, :, 224:].copy()  # CHW format [3, 224, 224]
    
    return sample

def gen_bc_dataset(num_samples, num_previous_strokes, num_after_strokes, base_dir, gen_true_img=False):
    curr_img_dir = os.path.join(base_dir, 'curr_img')
    goal_img_dir = os.path.join(base_dir, 'goal_img')
    action_dir = os.path.join(base_dir, 'action')
    true_img_dir = os.path.join(base_dir, 'true_img') if gen_true_img else None
    
    # 기존 디렉토리 제거
    print("Removing existing directories...")
    dirs_to_remove = [curr_img_dir, goal_img_dir, action_dir]
    if gen_true_img:
        dirs_to_remove.append(true_img_dir)
    for dir_path in dirs_to_remove:
        remove_dir(dir_path)
    
    # 디렉토리 생성
    print("Creating new directories...")
    dirs_to_create = [curr_img_dir, goal_img_dir, action_dir]
    if gen_true_img:
        dirs_to_create.append(true_img_dir)
    for dir_path in dirs_to_create:
        ensure_dir(dir_path)
    
    # 환경 초기화
    env = DrawingEnv()
    
    # 샘플 생성
    print(f"Generating {num_samples} samples with prev:{num_previous_strokes}, after:{num_after_strokes} strokes in current canvas...")
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # 샘플 생성
        sample = generate_sample(env, num_previous_strokes, num_after_strokes, gen_true_img)
        
        # 이미지 저장 (224x224 해상도)
        current_canvas_img = BezierDrawingCanvas()
        current_canvas_img.fill_image(Image.fromarray(np.transpose(sample["curr_img"], (1, 2, 0))))
        current_canvas_img.save_to_file(os.path.join(curr_img_dir, f'{i}.png'))
        
        target_img = BezierDrawingCanvas()
        target_img.fill_image(Image.fromarray(np.transpose(sample["goal_img"], (1, 2, 0))))
        target_img.save_to_file(os.path.join(goal_img_dir, f'{i}.png'))
        
        # true_img 저장 (gen_true_img가 True인 경우)
        if gen_true_img:
            true_img = BezierDrawingCanvas()
            true_img.fill_image(Image.fromarray(np.transpose(sample["true_img"], (1, 2, 0))))
            true_img.save_to_file(os.path.join(true_img_dir, f'{i}.png'))
        
        # 액션 저장
        save_action(sample["action"], os.path.join(action_dir, f'{i}.json'))
    
    print("Data generation completed!")

def main():
    parser = argparse.ArgumentParser(description='Generate imitation learning data using random strokes')
    parser.add_argument('--num_samples', type=int, required=True, help='Number of BC samples to generate')
    parser.add_argument('--num_previous_strokes', type=int, required=True, help='Number of strokes to draw in current canvas')
    parser.add_argument('--num_after_strokes', type=int, required=True, help='Number of strokes to draw in target canvas')
    parser.add_argument('--gen_true_img', action='store_true', help='Generate true_img (curr_img + 1 action)')
    
    args = parser.parse_args()
    
    # 저장 디렉토리 설정
    base_dir = './temp/bc'
    
    gen_bc_dataset(args.num_samples, args.num_previous_strokes, args.num_after_strokes, base_dir, args.gen_true_img)
    

if __name__ == "__main__":
    main() 