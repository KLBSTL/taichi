import cv2
import os

image_folder = 'image'  # 存放你的 PNG 图片的文件夹
video_name = 'cloth_sim.mp4'
fps = 30  # 每秒帧数

# 获取所有图片路径并排序
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  # 确保顺序正确

# 读取第一张图片获取大小
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 格式
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# 写入每一帧
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()
print(f'视频已保存为 {video_name}')
