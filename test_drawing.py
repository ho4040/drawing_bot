from drawing_env import BezierDrawingCanvas
import os
os.makedirs("./temp/test_drawing", exist_ok=True)

# BezierDrawingCanvas 인스턴스 생성
bezier_drawing = BezierDrawingCanvas() 
for i in range(10):
    bezier_drawing.draw_random_strokes(1)  # 랜덤 획 그리기 테스트
    bezier_drawing.save_to_file(f"./temp/test_drawing/drawing_{i}.png")  # 테스트 결과 저장