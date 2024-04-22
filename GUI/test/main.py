from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('GUI_v1.ui', self)  # 加载.ui文件

        # 设置默认启动页为第一页
        self.pages.setCurrentIndex(0)

        # 读取QComboBox的输入值
        self.sign_in_choose_value = self.Sign_in_choose.currentText()

        # 设置按钮的点击事件
        self.pushButton_2.clicked.connect(self.load_file_2)
        self.pushButton_3.clicked.connect(self.load_file_3)
        self.pushButton.clicked.connect(self.reset_buttons)
        self.sign_in.clicked.connect(self.go_to_page_2)

    def load_file_2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open file', '')
        if file_path:
            self.pushButton_2.setStyleSheet("background-color: green")
            self.pushButton_2.setToolTip(file_path)

    def load_file_3(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open file', '')
        if file_path:
            self.pushButton_3.setStyleSheet("background-color: green")
            self.pushButton_3.setToolTip(file_path)

    def reset_buttons(self):
        self.pushButton_2.setStyleSheet("")
        self.pushButton_2.setToolTip("")
        self.pushButton_3.setStyleSheet("")
        self.pushButton_3.setToolTip("")

    def go_to_page_2(self):
        self.pages.setCurrentIndex(1)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())