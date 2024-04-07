from PyQt6 import QtWidgets, uic
import sys

class Calculator(QtWidgets.QMainWindow):
    def __init__(self):
        super(Calculator, self).__init__()
        uic.loadUi('calculator.ui', self)

        self.button0 = self.findChild(QtWidgets.QPushButton, 'button0')
        self.button0.clicked.connect(self.digit_pressed)
        self.button1 = self.findChild(QtWidgets.QPushButton, 'button1')
        self.button1.clicked.connect(self.digit_pressed)
        self.buttonAdd = self.findChild(QtWidgets.QPushButton, 'buttonAdd')
        self.buttonAdd.clicked.connect(self.operation_pressed)
        self.buttonEqual = self.findChild(QtWidgets.QPushButton, 'buttonEqual')
        self.buttonEqual.clicked.connect(self.equal_pressed)

        self.display = self.findChild(QtWidgets.QLineEdit, 'display')

        self.current_operation = None
        self.current_value = 0

    def digit_pressed(self):
        sender = self.sender()
        new_value = int(sender.text())
        self.current_value = self.current_value * 10 + new_value
        self.display.setText(str(self.current_value))

    def operation_pressed(self):
        sender = self.sender()
        self.current_operation = sender.text()

    def equal_pressed(self):
        if self.current_operation == '+':
            self.current_value += int(self.display.text())
        self.display.setText(str(self.current_value))

app = QtWidgets.QApplication(sys.argv)
window = Calculator()
window.show()
app.exec()