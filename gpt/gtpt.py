import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QProgressBar, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
from ultralytics import YOLO
import numpy as np

class DefectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.model_path = r"E:\\university_class\\howtostudy\\temp\\detect-main\\NEU-DET\\neu_det_results\\train_run110\\weights\\best.pt"  # YOLOv8 模型路径
        self.model = YOLO(self.model_path)  # 加载模型

        self.initUI()

    def initUI(self):
        self.setWindowTitle('YOLOv8 缺陷检测')

        # 创建UI组件
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.detected_image_label = QLabel(self)
        self.detected_image_label.setAlignment(Qt.AlignCenter)

        self.select_button = QPushButton('选择图片', self)
        self.select_button.clicked.connect(self.select_image)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)

        self.log_area = QTextEdit(self)
        self.log_area.setReadOnly(True)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.detected_image_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_area)

        self.setLayout(layout)
        self.show()

    def select_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "",
                                                   "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        # 显示原始图片
        self.log_area.append(f"加载图片: {file_path}")
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.show_image(image_rgb, self.image_label)

        # 更新进度条
        self.progress_bar.setValue(30)

        # YOLOv8 检测
        self.log_area.append("正在进行检测...")
        results = self.model(file_path)  # 也可以传 numpy 图像: results = self.model(image_rgb)

        self.progress_bar.setValue(60)

        # 获取带标注的图片
        detected_image = results[0].plot()  # 返回 numpy RGB 图像
        self.show_image(detected_image, self.detected_image_label)

        # 更新进度条
        self.progress_bar.setValue(100)
        self.log_area.append("检测完成！")

    def show_image(self, image, label):
        # 确保是 RGB
        if image.shape[2] == 3:
            q_image = QImage(image.data, image.shape[1], image.shape[0], 3 * image.shape[1], QImage.Format_RGB888)
        else:
            # 处理灰度图
            q_image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DefectDetectionApp()
    sys.exit(app.exec_())
