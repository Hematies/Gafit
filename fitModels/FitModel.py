from energyTimeModels import EnergyTimeModel

class FitModel:
    data = dict()
    hyperparameters = dict()
    paramsMinMax = dict()

    def __init__(self, hyperparameters=dict(), data=dict(), paramsMinMax=dict()):
        self.hyperparameters = dict(hyperparameters)
        self.data = dict(data)
        self.paramsMinMax = dict(paramsMinMax)

    def getData(self):
        return dict(self.data)

    def setData(self, data):
        self.data = dict(data)

    def getHyperparameters(self):
        return dict(self.hyperparameters)

    def setHyperparameters(self, hyperparameters):
        self.hyperparameters = dict(hyperparameters)

    def getParamsMinMax(self):
        return dict(self.paramsMinMax)

    def setParamsMinMax(self, paramsMinMax):
        self.paramsMinMax = dict(paramsMinMax)

    def fit(self, model: EnergyTimeModel):
        '''
        Given an EnergyTimeModel, its parameters are fitted according to the
        implementation of FitModel. A dictionary with a set of results
        and metrics is returned:
        '''
        pass


    def computeFitness(self, model: EnergyTimeModel, params, areParamsNorm=False):
        pass
