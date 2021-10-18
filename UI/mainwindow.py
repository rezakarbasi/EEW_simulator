#%%
import sys
sys.path.append('./../')
sys.path.append('.')

from Data.gen_data.DataGeneration import DATA_GENERATOR
from Algorithm.PgaOptimization import PGA_OPTIMIZATION
from Algorithm.PgaOptimizationTorch import PGA_OPTIMIZOR_TORCH
from PyQt5.QtWidgets import QMainWindow, QComboBox, QApplication, QCheckBox, QPushButton, QRadioButton, QWidget,QAction, QHBoxLayout, QTabWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit
from Objects.objects import PARAMETER_TYPE,UI_OBJ

dataCombo = [DATA_GENERATOR()]
algCombo = [PGA_OPTIMIZATION()]#, PGA_OPTIMIZOR_TORCH()]

def getLog(settings:list=[],tosave:list=[],name:list=[],saveTag=''):
    import datetime
    import os

    thisTime = datetime.datetime.now()

    savePath = './result/' + saveTag + '--' + str(thisTime)[:-7]+'/'
    os.mkdir(savePath)

    settingStr = "settings : \ntime : "+str(thisTime)+"\nsave tag : "+saveTag+"\n\n"
    for s in settings:
        settingStr+=str(s)+'\n'

    with open(savePath+'settings.txt', 'w') as f:
        f.write(settingStr)
    
    for file,name in zip(tosave,name):
        if type(name)!=str:
            raise Exception("names must be str !!")
        
        with open(savePath+name, 'w') as f:
            f.write(file)
    
    return savePath


#%% plot the results
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import numpy as np

SPEED = 10 # frame/s

def PlotResults(dataset,signal,outTime ,outLat ,outLong, outC, outRec, savePath):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=SPEED, metadata=dict(artist='Me'), bitrate=1800)

    stations = []
    for station in dataset.stations:
        stations.append((station.place.lat,station.place.long))
    stations = np.array(stations)

    fig = plt.figure(figsize=(20, 10))

    #### map plot
    ax_map = fig.add_subplot(121)
    ax_map.set_aspect('equal')
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
        combo.currentIndexChanged.connect(self.GiveComboChange('data'))
        self.dataLayout.addRow("select data source : ", combo)
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
        self.algResText = QLabel("no config set")
        self.algLayout.addRow(self.algResText)
        
        self.setTabText(0,"Algorithm")
        self.algTab.setLayout(self.algLayout)

    def ResTabInit(self):
        self.resLayout = QFormLayout()
        self.resBtn = QPushButton("Generate Results")
        self.resLayout.addRow(self.resBtn)
        self.resBtn.clicked.connect(self.resBtnClick)
        self.resTab.setLayout(self.resLayout)
    
    def resBtnClick(self):
        dataset = dataCombo[self.dataSelected]
        algorithm = algCombo[self.algSelected]

        dataset.run()
        print('dataset ran s')
        signal = dataset.earthquake.signal
        outTime ,outLat ,outLong, outC, outRec = algorithm.run(dataset.stations)
        print('run successfully')
        savePath = getLog(settings=[dataset,algorithm],saveTag='') # TODO : tag must be get from the user
        PlotResults(dataset ,signal ,outTime ,outLat ,outLong, outC, outRec, savePath)
        # print(outTime ,outLat ,outLong, outC, outRec)
        pass

    def GiveComboChange(self,section): # section can be data or alg
        print('function created ',section)
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
        print('function called')

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


#%% get file dialogue
# from PyQt5.QtWidgets import QFileDialog
#         options = QFileDialog.Options()
#         options |= QFileDialog.DontUseNativeDialog
#         fileName, _ = QFileDialog.getOpenFileName(self,"select file", "","All Files (*);;Python Files (*.py)", options=options)
#         if fileName:
#             print(fileName)