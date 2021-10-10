# https://realpython.com/qt-designer-python/
# https://www.riverbankcomputing.com/static/Docs/PyQt6/
# https://nitratine.net/blog/post/how-to-import-a-pyqt5-ui-file-in-a-python-gui/
# https://www.blog.pythonlibrary.org/2018/05/30/loading-ui-files-in-qt-for-python/

from PyQt5 import QtWidgets, uic
import sys
sys.path.append('./..')

from Data.gen_data.DataGeneration import DATA_GENERATOR

dataCombo = [DATA_GENERATOR(),"hafez","shiraZ"]

def fillCombo(combo,items):
    for i in items:
        combo.addItem(str(i))

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.BTN.clicked.connect(self.onClick)
        
        fillCombo(self.DATA_COMBO, dataCombo)
        self.DATA_COMBO.setCurrentIndex(0)
        self.DATA_COMBO.currentIndexChanged.connect(self.on_combobox_changed)

        self.show()
    
    def on_combobox_changed(self,i=0):
        print(i)
        
    def onClick(self):
        print('clicked')

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()