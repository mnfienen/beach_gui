#!python

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../../python')


from PyQt4 import QtGui, QtCore
from modelbuilder import Ui_ModelBuilder
import model_script
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
        self.ui = Ui_ModelBuilder()
        self.ui.setupUi(self)
    
        #self.validate = True

        QtCore.QObject.connect(self.ui.pushButton_Execute, QtCore.SIGNAL("clicked()"), self.Execute )
        QtCore.QObject.connect(self.ui.pushButton_pickle, QtCore.SIGNAL("clicked()"), self.Wild)
        QtCore.QObject.connect(self.ui.pushButton_find, QtCore.SIGNAL("clicked()"), self.Find)


    def Find(self):
        data_path = QtGui.QFileDialog.getOpenFileName(self, caption="Open Data File", directory='../../data', filter='*.csv')
        self.ui.lineEdit_path.setText( data_path )
        
        
    def Wild(self):
        selected = self.ui.tableView.selectedIndexes()[0].row()
        

        method='pls'
        if self.ui.radioButton_PLS.isChecked(): method='pls'
        if self.ui.radioButton_Logistic.isChecked(): method='logistic'
        if self.ui.radioButton_Boost.isChecked(): method='boosting'

        model = model_script.Produce(self.result[selected], method, self.data_dict, self.target)
        
        outfile = 'model.mdl'
        o=open(outfile, 'w')
        model_pickler = pickle.dump(model, o)
        o.close()



 
    def Execute(self):
        method='pls'
        if self.ui.radioButton_PLS.isChecked(): method='pls'
        if self.ui.radioButton_Logistic.isChecked(): method='logistic'
        if self.ui.radioButton_Boost.isChecked(): method='boosting'

        threshold=[]
        if self.ui.checkBox_tProps.isChecked(): threshold.append(0)
        if self.ui.checkBox_tCounts.isChecked(): threshold.append(1)

        balance=[]
        if self.ui.checkBox_bFalseNeg.isChecked(): balance.append(0)
        if self.ui.checkBox_bFalseNegRate.isChecked(): balance.append(1)
        if self.ui.checkBox_bNDR.isChecked(): balance.append(2)
        
        infile = str( self.ui.lineEdit_path.text() )
        self.target = model_target = str( self.ui.lineEdit_target.text() ).lower()

        year = int( self.ui.lineEdit_year.text() )

        specificities = str( self.ui.lineEdit_specificities.text() )
        specificity = [float(limit) for limit in specificities.split(',')]

        [headers, result, self.data_dict] = model_script.Test(infile, model_target, method, year=year, specificity=specificity, balance=balance, threshold=threshold)
        self.result=result

        '''if len(result.squeeze().shape)==1: result=list( data )
        else: result = [list(row) for row in result.squeeze()]'''

        self.Populate_Table(result, headers)
        self.ui.pushButton_pickle.setEnabled( True )



    def Populate_Table(self, data, headers):
        # create the view
        tv = self.ui.tableView
        
        # set the table model
        tm = MyTableModel(data, headers, self) 
        tv.setModel(tm)
    
        # set the minimum size
        #tv.setMinimumSize(400, 300)
    
        # hide grid
        tv.setShowGrid(True)
    
        # set the font
        #font = QtGui.QFont("Courier New", 8)
        #tv.setFont(font)
    
        # set vertical header properties
        vh = tv.verticalHeader()
        vh.setVisible(True)
    
        # set horizontal header properties
        hh = tv.horizontalHeader()
        hh.setStretchLastSection(False)
    
        # set column width to fit contents
        tv.resizeColumnsToContents()
    
        # set row height
        #nrows = len(self.tabledata)
        #for row in xrange(nrows):
        #    tv.setRowHeight(row, 18)
    
        # enable sorting
        #tv.setSortingEnabled(True)
    
        return tv
 

class MyTableModel(QtCore.QAbstractTableModel): 
    def __init__(self, datain, headerdata, parent=None, *args): 
        """ datain: a list of lists
            headerdata: a list of strings
        """
        QtCore.QAbstractTableModel.__init__(self, parent, *args) 
        self.arraydata = datain
        self.headerdata = headerdata
 
    def rowCount(self, parent): 
        return len(self.arraydata) 
 
    def columnCount(self, parent): 
        return len(self.arraydata[0]) 
 
    def data(self, index, role): 
        if not index.isValid(): 
            return QtCore.QVariant() 
        elif role != QtCore.Qt.DisplayRole: 
            return QtCore.QVariant() 
        return QtCore.QVariant(self.arraydata[index.row()][index.column()]) 

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.headerdata[col])
        return QtCore.QVariant()

    def sort(self, Ncol, order):
        """Sort table by given column number.
        """
        self.emit(SIGNAL("layoutAboutToBeChanged()"))
        self.arraydata = sorted(self.arraydata, key=operator.itemgetter(Ncol))        
        if order == QtCore.Qt.DescendingOrder:
            self.arraydata.reverse()
        self.emit(SIGNAL("layoutChanged()"))





'''class MyWindow(QWidget): 
    def __init__(self, *args): 
        QWidget.__init__(self, *args) 

        # create table
        self.get_table_data()
        table = self.createTable() 
         
        # layout
        layout = QVBoxLayout()
        layout.addWidget(table) 
        self.setLayout(layout) 

    def get_table_data(self):
        stdouterr = os.popen4("dir c:\\")[1].read()
        lines = stdouterr.splitlines()
        lines = lines[5:]
        lines = lines[:-2]
        self.tabledata = [re.split(r"\s+", line, 4)
                     for line in lines]'''

if __name__ == "__main__":
    main()

