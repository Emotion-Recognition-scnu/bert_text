from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('GUI_v1.ui', self)  # 加载.ui文件

        # 设置默认启动页为第一页
        self.pages.setCurrentIndex(0)
        self.filepath1 = ""
        self.filepath2 = ""

        # 读取QComboBox的输入值
        self.sign_in_choose_value = self.Sign_in_choose.currentText()

        # 设置按钮的点击事件
        self.pushButton_2.clicked.connect(self.load_file_2)
        self.pushButton_3.clicked.connect(self.load_file_3)
        self.pushButton.clicked.connect(self.reset_buttons)
        self.sign_in.clicked.connect(self.go_to_page_2)

    def load_file_2(self):
        #global filepath1
        self.file_path1, _ = QFileDialog.getOpenFileName(self, 'Open file', '')
        if self.file_path1:
            self.pushButton_2.setStyleSheet("background-color: rgb(5,217,88)")
            self.pushButton_2.setText(self.file_path1)

    def load_file_3(self):
        #global filepath2
        self.file_path2, _ = QFileDialog.getOpenFileName(self, 'Open file', '')
        if self.file_path2:
            self.pushButton_3.setStyleSheet("background-color: rgb(5,217,88)")
            self.pushButton_3.setText(self.file_path2)

    def reset_buttons(self):
        self.pushButton_2.setStyleSheet("")
        self.pushButton_2.setText("未选择文件")
        self.pushButton_3.setStyleSheet("")
        self.pushButton_3.setText("未选择文件")
        self.file_path1 = ""
        self.file_path2 = ""

    def go_to_page_2(self):
        print(self.file_path1)
        print(self.file_path2)
        if self.filepath1 == '' and self.filepath2 == '':
            return
        self.pages.setCurrentIndex(1)
        self.display_page_1()

    def display_page_1(self):
        self.order_info.setText(
            "分析模式：\n" + self.sign_in_choose_value + "\n" + "文件1：" + self.filepath1 + "\n" + "文件2：" + self.filepath2)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
