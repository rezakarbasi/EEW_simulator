#%%
import sys

from numpy.lib.npyio import save

sys.path.append('./../')
sys.path.append('.')

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from Data.gen_data.DataGeneration import DATA_GENERATOR
from Data.real_data.LoadData import LOAD_REAL_DATA
from Algorithm.PgaOptimization import PGA_OPTIMIZATION
from Algorithm.PgaOptimizationTorch import PGA_OPTIMIZOR_TORCH
from PyQt5.QtWidgets import QMainWindow, QComboBox, QApplication, QCheckBox, QPushButton, QRadioButton, QWidget,QAction, QHBoxLayout, QTabWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit, QMessageBox, QGridLayout
from Objects.objects import PARAMETER_TYPE, PLACES,UI_OBJ
from Functions import FindDist
from PyQt5.QtWidgets import QFileDialog

dataCombo = [DATA_GENERATOR(),LOAD_REAL_DATA()]
algCombo = [PGA_OPTIMIZATION(), PGA_OPTIMIZOR_TORCH()]

def getLog(settings:list=[],tosave:list=[],names:list=[],saveTag=''):
    import datetime
    import os

    thisTime = datetime.datetime.now()

    savePath = './result/' + saveTag + '--' + str(thisTime)[:-7]+'/'
    os.mkdir(savePath)

    settingStr = "settings : \ntime : "+str(thisTime)+"\nsave tag : "+saveTag+"\n\n"
    for s in settings:
        settingStr+=s.GetConfigStr()+'\n\n'

    with open(savePath+'settings.txt', 'w') as f:
        f.write(settingStr)
    
    for file,name in zip(tosave,names):
        if type(name)!=str:
            raise Exception("names must be str !!")
        
        with open(savePath+name, 'w') as f:
            f.write(file)
    
    return savePath


#%% plot the results
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib import animation

SPEED = 10 # frame/s

def PlotError(dataset,signal,outTime ,outLat ,outLong, outC, outRec, savePath):
    err = []
    for lat,long in zip(outLat,outLong):
        err.append(PLACES.distance(dataset.Give_Center(),lat,long))
    plt.close()
    plt.figure()
    plt.plot(outTime,err)
    plt.yscale('log')
    plt.ylim([0.1,60])
    plt.title('error in steps (km)')
    plt.xlabel('running steps')
    plt.ylabel('error in result')
    plt.savefig(savePath+'error.png')
    plt.close()

def PlotResults(dataset,signal,outTime ,outLat ,outLong, outC, outRec, savePath):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=SPEED, metadata=dict(artist='Me'), bitrate=1800)

    stations = []
    for station in dataset.stations:
        stations.append((station.place.lat,station.place.long))
    stations = np.array(stations)

    # print('stations : \n',stations)

    mean = np.mean(stations,axis=0)
    maxx = np.max(stations,axis=0)

    deltaLat = FindDist(mean[0],mean[1],20,'lat')
    deltaLon = FindDist(mean[0],mean[1],20,'long')

    fig = plt.figure(figsize=(20, 10))

    #### map plot
    ax_map = fig.add_subplot(121)
    
    # ax_map.set_aspect('equal')
    ax_map.set_aspect(deltaLon/deltaLat)
    print(maxx,deltaLat)
    ax_map.add_patch(patches.Rectangle((maxx[0]-deltaLat-0.05,maxx[1]-0.05-deltaLat/10), deltaLat,deltaLat/10,color="black"))
    ax_map.text(maxx[0]-deltaLat*2/3-0.05,maxx[1]-0.04, '20KM', fontsize = 8)

    ax_map.set(xlim=(min(stations[:,0])-0.1, max(stations[:,0])+0.1), ylim=(min(stations[:,1]-0.1), max(stations[:,1])+0.1))
    scat = ax_map.scatter([0],[0],label='predicted')
    ax_map.scatter([dataset.earthquake.place.lat],[dataset.earthquake.place.long],s=500,marker='*',c='r',label='epicenter')
    # ax.scatter(stations[:,0],stations[:,1],marker='^',c='k',label='stations')
    scatStation = ax_map.scatter([],[],marker='^',c='k',label='stations')
    ax_map.set_xlabel('latitude')
    ax_map.set_ylabel('longitude')
    line = ax_map.plot([], [], color='b', lw=1, alpha=0.5)[0]
    ax_map.legend()
    #### END map plot

    #### signal plots
    signalLines = []
    for i,key in enumerate(signal):
        i += 1
        sig = signal[key]
        ax = fig.add_subplot(len(signal),2,i*2)
        # ax.set_aspect('equal')
        minn , maxx = min(sig),max(sig)
        dist = maxx-minn
        ax.set(xlim=(0, len(sig)), ylim=(minn-abs(dist*0.1),maxx+abs(dist*0.1)))
        if i != len(signal):
            ax.xaxis.set_visible(False)
        ax.set_title(key)
        signalLines.append(ax.plot([], [], color='b', lw=1, alpha=0.5)[0])
    #### END signal plots


    def animate(i):
        global textsIn
        
        #### map plot
        #predicted
        scat.set_offsets(np.array((outLat[i],outLong[i])))
        # set texts
        for t in textsIn:
            t.set_visible(False)
        textsIn = []
        for s in outRec[i]:
            textsIn.append(ax_map.text(s['place'].lat, s['place'].long-0.03, '{:2.2f}'.format(s['pga']),ha='center', 
                                va='center',fontsize=8))
        # visible stations
        showStations  = np.array([(s['place'].lat,s['place'].long) for s in outRec[i]])
        if len(showStations)>0:
            scatStation.set_offsets(np.c_[showStations[:,0], showStations[:,1]])
        else :
            scatStation.set_offsets(np.c_[[],[]])
        
        ax_map.set_title(str(outTime[i])+'\nC = '+str(outC[i]))
        line.set_xdata(outLat[:i+1])
        line.set_ydata(outLong[:i+1])
        #### END map plot
        
        #### signal plots
        for idx,signalLine in enumerate(signalLines):
            key = list(signal)[idx]
            sig = signal[key]
            if len(sig)>i:
                signalLine.set_ydata(sig[:i])
                signalLine.set_xdata(range(i))
        #### END signal plots
    
    global textsIn
    textsIn = []

    anim = FuncAnimation(fig, animate, interval=int(1050/SPEED), frames=len(outTime))
    anim.save(savePath + 'sim.mp4', writer=writer)
    fig.show()

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
        self.resLayout = QFormLayout()
        self.saveTag = QLineEdit()
        self.resLayout.addRow("Saving Tag : ",self.saveTag)
        self.resBtn = QPushButton("Generate Results")
        self.resLayout.addRow(self.resBtn)
        self.resBtn.clicked.connect(self.resBtnClick)
        self.resTab.setLayout(self.resLayout)
    
    def resBtnClick(self):
        if self.dataSelected==None or self.algSelected == None:
            showdialog()
            return 
        dataset = dataCombo[self.dataSelected]
        algorithm = algCombo[self.algSelected]
        algorithm.reset()

        dataset.run()
        print('dataset ran s')
        signal = dataset.earthquake.signal
        outTime ,outLat ,outLong, outC, outRec = algorithm.run(dataset.stations)
        print('run successfully')
        print('save tag is : ',self.saveTag.text())
        savePath = getLog(settings=[dataset,algorithm],saveTag=self.saveTag.text())
        PlotResults(dataset ,signal ,outTime ,outLat ,outLong, outC, outRec, savePath)
        PlotError(dataset ,signal ,outTime ,outLat ,outLong, outC, outRec, savePath)
        # print(outTime ,outLat ,outLong, outC, outRec)
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
                btn.clicked.connect(self.FillTheParamtersFunction(self.inputTexts[-1]))

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

    def FillTheParamtersFunction(self,lineEdit):
        def function():
            lineEdit.setText(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
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
