from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('GUI_v1.ui', self)

        self.pages.setCurrentIndex(0)
        self.file_path1 = ""
        self.file_path2 = ""
        self.status = 0

        self.pushButton_2.clicked.connect(self.load_file_2)
        self.pushButton_3.clicked.connect(self.load_file_3)
        self.pushButton.clicked.connect(self.reset_buttons)
        self.sign_in.clicked.connect(self.go_to_page_2)
        self.to_select_page.clicked.connect(self.return_to_page_1)
        self.to_payment_details.clicked.connect(self.to_page_3)
        self.to_manage_room_page_2.clicked.connect(self.go_to_page_2)

    def load_file_2(self):
        self.file_path1, _ = QFileDialog.getOpenFileName(self, 'Open file', '')
        if self.file_path1:
            self.pushButton_2.setStyleSheet("""
            QPushButton {
                background-color: rgb(5,217,88);
            }
            QPushButton:hover {
                background-color: rgb(0,180,0);
            }
        """)
            self.pushButton_2.setText(self.file_path1)

    def load_file_3(self):
        self.file_path2, _ = QFileDialog.getOpenFileName(self, 'Open file', '')
        if self.file_path2:
            self.pushButton_3.setStyleSheet("""
            QPushButton {
                background-color: rgb(5,217,88);
            }
            QPushButton:hover {
                background-color: rgb(0,180,0);
            }
        """)
            self.pushButton_3.setText(self.file_path2)

    def reset_buttons(self):
        self.pushButton_2.setStyleSheet("""QPushButton:hover {
                background-color: rgb(160,160,160);
            }""")
        self.pushButton_2.setText("未选择文件")
        self.pushButton_3.setStyleSheet("""QPushButton:hover {
                background-color: rgb(160,160,160);
            }""")
        self.pushButton_3.setText("未选择文件")
        self.file_path1 = ""
        self.file_path2 = ""

    def go_to_page_2(self):
        print(self.file_path1)
        print(self.file_path2)
        self.pages.setCurrentIndex(1)
        self.display_page_1()

    def display_page_1(self):
        self.sign_in_choose_value = self.Sign_in_choose.currentText()
        self.status = 0
        if ((self.sign_in_choose_value == "文本单模态" and self.file_path1 != "") or
                (self.sign_in_choose_value == "自动" and self.file_path1 != "" and self.file_path2 == "")):
            self.order_info.setText(
                "分析模式：" + self.sign_in_choose_value + "\n" +
                "文本文件：" + self.file_path1)
        elif ((self.sign_in_choose_value == "视频单模态" and self.file_path1 != "") or
              (self.sign_in_choose_value == "自动" and self.file_path2 != "" and self.file_path1 == "")):
            self.order_info.setText(
                "分析模式：" + self.sign_in_choose_value + "\n" +
                "视频文件：" + self.file_path2)
        elif (self.sign_in_choose_value == "多模态" or '自动') and self.file_path1 != "" and self.file_path2 != "":
            self.order_info.setText(
                "分析模式：" + self.sign_in_choose_value + "\n" +
                "文本文件：" + self.file_path1 + "\n" +
                "视频文件：" + self.file_path2)
        else:
            self.order_info.setText("未选择文件或分析模式有误")
            self.status = 1

    def return_to_page_1(self):
        if self.status == 0:
            self.reset_buttons()
        self.pages.setCurrentIndex(0)
        self.order_info.setText("")

    def to_page_3(self):
        if self.status == 1:
            return
        self.pages.setCurrentIndex(3)
        self.comboBox.setCurrentIndex(0)
        self.comboBox_2.setCurrentIndex(0)
        self.room_type.setCurrentIndex(0)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
