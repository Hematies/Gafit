from energyTimeModels.EnergyTimeModel import EnergyTimeModel
from fitModels.FitModel import FitModel

class Fit:

    '''
    def __init__(self):
        self.data: dict = dict()
        self.energyTimeModel: EnergyTimeModel = EnergyTimeModel()
        self.fitModel: FitModel = FitModel()
    '''

    def __init__(self, data: dict = dict(), energyTimeModel: EnergyTimeModel = EnergyTimeModel(),
                 fitModel: FitModel = FitModel()):
        self.data: dict = dict(data)
        self.energyTimeModel: EnergyTimeModel = energyTimeModel
        self.fitModel: FitModel = fitModel

    def fit(self):
        '''
        We return the stats collected by the FitModel object when fitting:
        '''
        return self.fitModel.fit(self.energyTimeModel)

    def getEnergyTimeModel(self):
        return self.energyTimeModel

    def getFitModel(self):
        return self.fitModel

    def getData(self):
        return dict(self.data)

    def setEnergyTimeModel(self, model: EnergyTimeModel):
        pass

    def setFitModel(self, model: FitModel):
        pass

    def setData(self, data: dict):
        pass
