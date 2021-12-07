#%%
import sys
from PyQt5.QtCore import QLine

from numpy.lib.npyio import save

sys.path.append('./../')
sys.path.append('.')

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from Data.gen_data.DataGeneration import DATA_GENERATOR
from Data.gen_data.StepGenerator import STEP_GENERATOR
from Data.gen_data.DataGenerationFromFile import DATA_GENERATOR_FROM_FILE
from Data.real_data.LoadData import LOAD_REAL_DATA
from Algorithm.PgaOptimization import PGA_OPTIMIZATION
from Algorithm.PgaOptimizationTorch import PGA_OPTIMIZOR_TORCH
from Algorithm.StaLtaLoc import STA_LTA_LOCATION
from PyQt5.QtWidgets import QMainWindow, QComboBox, QApplication, QCheckBox, QPushButton, QRadioButton, QWidget,QAction, QHBoxLayout, QTabWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit, QMessageBox, QGridLayout
from Objects.objects import PARAMETER_TYPE, PLACES,UI_OBJ
from Functions import FindDist, PlotError, PlotResults, MakePath, GetLog
from PyQt5.QtWidgets import QFileDialog
import matplotlib
# matplotlib.use("TkAgg")
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

dataCombo = [STEP_GENERATOR(),DATA_GENERATOR(),LOAD_REAL_DATA(),DATA_GENERATOR_FROM_FILE()]
algCombo = [PGA_OPTIMIZATION(), PGA_OPTIMIZOR_TORCH(),STA_LTA_LOCATION()]
plotFPS = 10


def StringIsFloat(string:str):
    try:
        return float(string)
    except:
        return 0
 
#%% open a message box
def showdialog():
   msg = QMessageBox()
   msg.setIcon(QMessageBox.Warning)

   msg.setText("you didn't enter all the parameters")
   msg.setInformativeText("refer to algorithm and data section")
   msg.setWindowTitle("Input Warning")
#    msg.setDetailedText("The details are as follows:")
   msg.setStandardButtons(QMessageBox.Ok)
#    msg.buttonClicked.connect(msgbtn)
	
   retval = msg.exec_()

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

        self.dataSelected = None
        self.algSelected = None
    		
    def DataTabInit(self):
        self.dataLayout = QFormLayout()
        combo = QComboBox()
        combo.addItem('select')
        for i in dataCombo:
            combo.addItem(str(i))
        combo.currentIndexChanged.connect(self.GiveComboChange('data'))
        self.dataLayout.addRow("select data source : ", combo)

        btn = QPushButton("Reset Parameters")
        btn.clicked.connect(self.GiveResetBtnClick('data'))
        self.dataLayout.addRow(btn)

        self.dataResText = QLabel("no config set")
        self.dataLayout.addRow(self.dataResText)
        
        self.setTabText(0,"Data")
        self.dataTab.setLayout(self.dataLayout)
    		
    def AlgorithmTabInit(self):
        self.algLayout = QFormLayout()
        combo = QComboBox()
        combo.addItem('select')
        for i in algCombo:
            combo.addItem(str(i))
        combo.currentIndexChanged.connect(self.GiveComboChange('alg'))
        self.algLayout.addRow("select Algorithm : ", combo)

        btn = QPushButton("Reset Parameters")
        btn.clicked.connect(self.GiveResetBtnClick('alg'))
        self.algLayout.addRow(btn)

        self.algResText = QLabel("no config set")
        self.algLayout.addRow(self.algResText)
        
        self.setTabText(1,"Algorithm")
        self.algTab.setLayout(self.algLayout)

    def ResTabInit(self):
        self.targetLatLabel = QLineEdit()
        self.targetLonLabel = QLineEdit()

        localLayout = QGridLayout()
        localLayout.addWidget(QLabel('target latitude : '), 0, 0)
        localLayout.addWidget(self.targetLatLabel, 0, 1)
        localLayout.addWidget(QLabel('\ttarget longitude :'), 0, 2)
        localLayout.addWidget(self.targetLonLabel, 0, 3)

        self.resLayout = QFormLayout()
        self.saveTag = QLineEdit()
        self.resLayout.addRow("Saving Tag : ",self.saveTag)
        self.frameText = QLineEdit()
        self.resLayout.addRow("Frame Rate : ",self.frameText)
        self.resBtn = QPushButton("Generate Results")
        self.resLayout.addRow(localLayout)
        self.resLayout.addRow(self.resBtn)
        self.resBtn.clicked.connect(self.resBtnClick)
        self.resTab.setLayout(self.resLayout)
    
    def resBtnClick(self):
        if self.dataSelected==None or self.algSelected == None:
            showdialog()
            return 
        if self.frameText.text().isdigit():
            global plotFPS
            plotFPS = int(self.frameText.text())

        dataset = dataCombo[self.dataSelected]
        algorithm = algCombo[self.algSelected]
        algorithm.reset()

        dataset.run()

        targetLat = StringIsFloat(self.targetLatLabel.text())
        targetLon = StringIsFloat(self.targetLonLabel.text())

        if targetLat==0 and targetLon==0:
            targetLat = dataset.stations[0].place.lat
            targetLon = dataset.stations[0].place.long

        self.targetPlace = PLACES(targetLat,targetLon)

        saveTag=self.saveTag.text()

        print('dataset ran s')
        signal = dataset.earthquake.signal
        outTime ,outLat ,outLong, outC, outRec, warn, spendingTime = algorithm.run(dataset.stations,self.targetPlace)
        print('run successfully')
        print('save tag is : ',saveTag)

        savePath = MakePath(saveTag=saveTag)
        PlotResults(dataset ,signal ,outTime ,outLat ,outLong, outC, outRec, savePath, warn, plotFPS)
        err = PlotError(dataset ,signal ,outTime ,outLat ,outLong, outC, outRec, savePath)
        GetLog(savePath=savePath,settings=[dataset,algorithm],saveTag=saveTag,saveString="last error : {}".format(err[-1]),
                toSaves=[err,spendingTime],names=['error.pickle','spending.pickle'])
        pass

    def GiveResetBtnClick(self,section):
        # print('reset function created ',section)
        def resetClick():
            if section == 'alg' :
                idx = self.algSelected
                combo = algCombo
                self.selectedResText = self.algResText
            else :
                idx = self.dataSelected
                combo = dataCombo
                self.selectedResText = self.dataResText

            self.selectedCombo = combo
            self.selectedIdx = idx

            try :
                if len(combo[idx].parameters)==0 or idx<0:
                    self.cleanTheBox()
                    return 
            except:
                self.cleanTheBox()
                return 

            self.w = SecondWindow(combo[idx].parameters,self)
            self.w.show()
            self.hide()
        
        return resetClick

    def GiveComboChange(self,section): # section can be data or alg
        # print('function created ',section)
        def comboChange(i):
            counter = 0
            if i!=0:
                counter = i-1

                if section == 'alg' :
                    self.algSelected = counter
                    combo = algCombo
                    self.selectedResText = self.algResText
                else :
                    self.dataSelected = counter
                    combo = dataCombo
                    self.selectedResText = self.dataResText

                self.selectedCombo = combo
                self.selectedIdx = counter

                try :
                    if len(combo[counter].parameters)==0 or counter<0:
                        self.cleanTheBox()
                        return 
                except:
                    self.cleanTheBox()
                    return 

                self.w = SecondWindow(combo[counter].parameters,self)
                self.w.show()
                self.hide()
        
        return comboChange

    
    def cleanTheBox(self):
        self.selectedResText.setText("no config set")

    def ReturnFromWindow(self):
        self.show()
        self.w.close()
        printed = (self.selectedCombo[self.selectedIdx].__repr__())
        # print(printed)
        self.selectedResText.setText(printed)
        

#%% Secondary Window
class SecondWindow(QWidget):
    def __init__(self,parametersToGet:list,parent):
        super().__init__()
        layout = QFormLayout()
        self.parent = parent

        self.pramameters = parametersToGet
        self.inputTexts = []
        for p in self.pramameters:
            if p.openPathFinder:
                self.inputTexts.append(QLineEdit())
                self.inputTexts[-1].setPlaceholderText(p.help)

                btn = QPushButton("...")
                btn.clicked.connect(self.FillTheParamtersFunction(self.inputTexts[-1],'Directory'))

                localLayout = QGridLayout()
                localLayout.addWidget(QLabel(p.name), 0, 0)
                localLayout.addWidget(self.inputTexts[-1], 0, 1)
                localLayout.addWidget(btn, 0, 2)

                layout.addRow(localLayout)

            elif p.openFileFinder:
                self.inputTexts.append(QLineEdit())
                self.inputTexts[-1].setPlaceholderText(p.help)

                btn = QPushButton("...")
                btn.clicked.connect(self.FillTheParamtersFunction(self.inputTexts[-1],'File'))

                localLayout = QGridLayout()
                localLayout.addWidget(QLabel(p.name), 0, 0)
                localLayout.addWidget(self.inputTexts[-1], 0, 1)
                localLayout.addWidget(btn, 0, 2)

                layout.addRow(localLayout)

            else:
                self.inputTexts.append(QLineEdit())
                layout.addRow(p.name,self.inputTexts[-1])
                self.inputTexts[-1].setPlaceholderText(p.help)
        
        btn = QPushButton("click")
        btn.clicked.connect(self.btnClick)
        layout.addRow(btn)
        self.setLayout(layout)

    def FillTheParamtersFunction(self,lineEdit,whatToGet='Dir'):
        def function():
            if 'dir' in whatToGet.lower():
                lineEdit.setText(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
            else :
                a=QFileDialog.getOpenFileName()[0]
                lineEdit.setText(str(a))
                print(a)
        return function
    
    def btnClick(self):
        for param,inText in zip(self.pramameters,self.inputTexts):
            text = inText.text()
            try:
                param.setValue(text)
                inText.setStyleSheet("""QLineEdit { background-color: white; color: black }""")
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
