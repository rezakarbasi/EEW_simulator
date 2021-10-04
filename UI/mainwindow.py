# link to connect ui file to python
# https://www.blog.pythonlibrary.org/2018/05/30/loading-ui-files-in-qt-for-python/

from PyQt6 import uic
from PyQt6.QtWidgets import QApplication

Form, Window = uic.loadUiType("untitled.ui")

app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.show()
app.exec()
