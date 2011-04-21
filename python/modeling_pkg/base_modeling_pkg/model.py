class Model(object):
    '''Generic base class for models, containing some basic methods'''

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def CV(self, folds=0, **args):
        pass

    def MSEP(self, ncomp, **args):
        pass

    def Predict(self, prediction_frame):
        pass