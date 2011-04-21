#!python

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../../python')


from PyQt4 import QtGui, QtCore
from modeluser import Ui_ModelUser
import numpy as np
import pickle



def main(): 
    app = QtGui.QApplication(sys.argv)
    f = MyForm()
    f.show()
    sys.exit(app.exec_())





class MyForm(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_ModelUser()
        self.ui.setupUi(self)
    
        #self.validate = True

        QtCore.QObject.connect(self.ui.pushButton_Wild, QtCore.SIGNAL("clicked()"), self.SelectFile )
        QtCore.QObject.connect(self.ui.pushButton_predict, QtCore.SIGNAL("clicked()"), self.Predict )
        
        
    def SelectFile(self):
        model_file = QtGui.QFileDialog.getOpenFileName(self, caption="Open Model", directory='../', filter='*.mdl')
        
        model_file = open(model_file, 'r')
        self.model = pickle.load(model_file)
        model_file.close()

 
    
    def Predict(self):
        heads = ['year','julian','NTU','Rain_24','Rain_48','Total_X100','MGD','AVG_WATER_TEMP','WAVE_HEIGHT','AVG_AIR_TEMP','WIND_SPEED','LAKELEVEL_MLWK','X3qtr']

        heads = [head.lower() for head in heads]

        inputs = [float(self.ui.lineEdit_year.text()), float(self.ui.lineEdit_julian.text()), float(self.ui.lineEdit_turbidity.text()), float(self.ui.lineEdit_24_precip.text()), float(self.ui.lineEdit_48_precip.text()), float(self.ui.lineEdit_flow_total.text()), float(self.ui.lineEdit_flow_mgd.text()), float(self.ui.lineEdit_water_temp.text()), float(self.ui.lineEdit_wave_height.text()), float(self.ui.lineEdit_air_temp.text()), float(self.ui.lineEdit_wind.text()), float(self.ui.lineEdit_level.text()), float(self.ui.lineEdit_x3qtr.text())]
        
        prediction_dict = dict(zip(heads, inputs))
        
        output = self.model.Predict(prediction_dict)        
        self.ui.label_output.setText( str(output) )




if __name__ == "__main__":
    main()

