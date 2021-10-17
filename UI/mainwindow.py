# https://realpython.com/qt-designer-python/
# https://www.riverbankcomputing.com/static/Docs/PyQt6/
# https://nitratine.net/blog/post/how-to-import-a-pyqt5-ui-file-in-a-python-gui/
# https://www.blog.pythonlibrary.org/2018/05/30/loading-ui-files-in-qt-for-python/

# from PyQt5 import QtWidgets, uic
# import sys
# sys.path.append('./..')

# from Data.gen_data.DataGeneration import DATA_GENERATOR

# dataCombo = [DATA_GENERATOR(),"hafez","shiraZ"]

# def fillCombo(combo,items):
#     for i in items:
#         combo.addItem(str(i))

# class Ui(QtWidgets.QMainWindow):
#     def __init__(self):
#         super(Ui, self).__init__()
#         uic.loadUi('mainwindow.ui', self)
#         self.BTN.clicked.connect(self.onClick)
        
#         fillCombo(self.DATA_COMBO, dataCombo)
#         self.DATA_COMBO.setCurrentIndex(0)
#         self.DATA_COMBO.currentIndexChanged.connect(self.on_combobox_changed)

#         self.show()
    
#     def on_combobox_changed(self,i=0):
#         print(i)
        
#     def onClick(self):
#         print('clicked')

# app = QtWidgets.QApplication(sys.argv)
# window = Ui()
# app.exec_()

#%%
import sys
sys.path.append('./../')
sys.path.append('.')

from Data.gen_data.DataGeneration import DATA_GENERATOR
from PyQt5.QtWidgets import QMainWindow, QComboBox, QApplication, QCheckBox, QPushButton, QRadioButton, QWidget,QAction, QHBoxLayout, QTabWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit
from Objects.objects import PARAMETER_TYPE,UI_OBJ

dataCombo = [DATA_GENERATOR(),"hafez","shiraZ"]

#%% MainWindow class
class MainWindow(QTabWidget):
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self.dataTab = QWidget()
        self.algTab = QWidget()
        self.resTab = QWidget()
        		
        self.addTab(self.dataTab,"Data")
        self.addTab(self.algTab,"Algorithm")
        self.addTab(self.resTab,"Result")
        self.DataTabInit()
        self.AlgorithmTabInit()
        self.ResTabInit()
        self.setWindowTitle("earthquake simulator app")
    		
    def DataTabInit(self):
        self.dataLayout = QFormLayout()
        combo = QComboBox()
        combo.addItem('select')
        for i in dataCombo:
            combo.addItem(str(i))
        combo.currentIndexChanged.connect(self.DataComboChange)
        self.dataLayout.addRow("select data source : ", combo)
        self.dataResText = QLabel("no config set")
        self.dataLayout.addRow(self.dataResText)
        
        self.setTabText(0,"Data")
        self.dataTab.setLayout(self.dataLayout)
    		
    def AlgorithmTabInit(self):
        layout = QFormLayout()
        sex = QHBoxLayout()
        sex.addWidget(QRadioButton("Male"))
        sex.addWidget(QRadioButton("Female"))
        layout.addRow(QLabel("Sex"),sex)
        layout.addRow("Date of Birth",QLineEdit())
        self.setTabText(1,"Algorithm")
        self.algTab.setLayout(layout)
        		
    def ResTabInit(self):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("subjects"))
        layout.addWidget(QCheckBox("Physics"))
        layout.addWidget(QCheckBox("Maths"))
        self.setTabText(2,"Result")
        self.resTab.setLayout(layout)
          
    def DataComboChange(self,i):
        if i!=0:
            self.dataSelected = i-1
            self.w = SecondWindow(dataCombo[self.dataSelected].parameters,self)
            self.w.show()
            self.hide()
    
    def ReturnFromWindow(self):
        self.show()
        self.w.close()
        printed = (dataCombo[self.dataSelected].__repr__())
        print(printed)
        self.dataResText.setText(printed)
        

#%% Secondary Window
class SecondWindow(QWidget):
    def __init__(self,parametersToGet:list,parent):
        super().__init__()
        layout = QFormLayout()
        self.parent = parent

        self.pramameters = parametersToGet
        self.inputTexts = []
        for p in self.pramameters:
            self.inputTexts.append(QLineEdit())
            layout.addRow(p.name,self.inputTexts[-1])
            self.inputTexts[-1].setPlaceholderText(p.help)
        
        btn = QPushButton("click")
        btn.clicked.connect(self.btnClick)
        layout.addRow(btn)
        self.setLayout(layout)
    
    def btnClick(self):
        for param,inText in zip(self.pramameters,self.inputTexts):
            text = inText.text()
            try:
                param.setValue(text)
            except:
                inText.setStyleSheet("""QLineEdit { background-color: red; color: black }""")
                return
        self.parent.ReturnFromWindow()

    def closeEvent(self, event):
        self.parent.ReturnFromWindow()
        event.accept()

#%% main function

def main():
   app = QApplication(sys.argv)
   ex = MainWindow()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()