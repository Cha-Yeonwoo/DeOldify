import fastai
from deoldify import device
from deoldify.device_id import DeviceId
# GPU가 있다면 GPU를 사용 설정
device.set(device=DeviceId.GPU0)

from deoldify.visualize import *
import glob
import os

# 서버 경로 설정
server_image_folder = '/mnt/disk1/ivymm02/images/train/angry'
output_folder = '/mnt/disk1/ivymm02/new_images/angry'

# DeOldify 모델 로드
colorizer = get_image_colorizer(artistic=True)

# 서버의 모든 이미지 파일을 순회
for image_path in glob.glob(f'{server_image_folder}/*.jpg'):  # 확장자에 따라 조정 필요
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    # 이미지 컬러화 실행
    colorizer.plot_transformed_image(image_path, render_factor=10, display_render_factor=True, figsize=(256,256), results_dir=output_folder)
    print(f'Processed and saved to {output_path}')
