import numpy as np

class EnergyTimeModel:

    data: dict = dict()
    hyperparameters: dict = dict()
    paramsMinMax: dict = dict()

    def __init__(self, hyperparameters = dict(), data = dict(), paramsMinMax = dict()):
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

    def normalizeParams(self, params):
        assert len(self.paramsMinMax) > 0
        res = dict()
        for k,v in params.items():
            min_, max_ = self.paramsMinMax[k]
            # If the min. or max. is indicated as parameter, we
            # we will set the min. or max. as the value of such parameter:
            if min_ in self.paramsMinMax.keys():
                min_ = params[min_]
            if max_ in self.paramsMinMax.keys():
                max_ = params[max_]
            assert min_ <= v <= max_
            res[k] = (v - min_) /(max_ - min_)
        return res

    def denormalizeParams(self, params):
        assert len(self.paramsMinMax) > 0
        res = dict()
        reassignments = dict()

        if type(params) is dict:
            for k, v in params.items():
                min_, max_ = self.paramsMinMax[k]

                # If the min. or max. is indicated as parameter, we
                # we will set the min. or max. as the value of such parameter:
                if min_ in self.paramsMinMax.keys() or max_ in self.paramsMinMax.keys():
                    reassignments[k] = v
                else:
                    res[k] = v * (max_ - min_) + min_
                    # assert min_ <= res[k] <= max_
        else:
            params_ = params# list(params)
            for i in range(0, len(params_)):
                k = list(self.paramsMinMax.keys())[i]
                v = params_[i]
                min_, max_ = self.paramsMinMax[k]

                # If the min. or max. is indicated as parameter, we
                # we will set the min. or max. as the value of such parameter:
                if min_ in self.paramsMinMax.keys() or max_ in self.paramsMinMax.keys():
                    reassignments[k] = v
                else:
                    res[k] = v * (max_ - min_) + min_
                     #assert min_ <= res[k] <= max_

        # We proceed to the reassignments until none are to be performed:
        while len(reassignments) > 0:
            deleteList = []
            for k, v in reassignments.items():
                min_, max_ = self.paramsMinMax[k]
                if min_ in self.paramsMinMax.keys():
                    if min_ in res:
                        min_ = res[min_]
                    else:
                        continue
                if max_ in self.paramsMinMax.keys():
                    if max_ in res:
                        max_ = res[max_]
                    else:
                        continue
                # assert min_ <= v <= max_
                res[k] = v * (max_ - min_) + min_
                deleteList.append(k)
            for k in deleteList:
                reassignments.pop(k)
        return res

    '''
    def computeFitness(self, params, areParamsNorm = False):
        if areParamsNorm:
            return self.computeFitness(self.denormalizeParams(params), False)
        else:
            pass
    '''

    def computeEnergyTimeCosts(self, params):
        '''
        Two tables (lists, dataframes, matrices...) are returned:
        - "energy" costs table (one cost per sample)
        - "time" costs table (one cost per sample)
        '''
        pass

    def getRealEnergyTimeCosts(self):
        '''
        Two tables (lists, dataframes, matrices...) are returned:
        - "energy" costs table (one cost per sample)
        - "time" costs table (one cost per sample)
        '''
        pass

    def computeEnergyTimeErrors(self, params):
        '''
        Two float values are to be returned:
        - Energy RMSE.
        - Time RMSE.
        '''
        pass

    '''
    def computeEnergyTimeNormalizedErrors(self, params):
        # Two float values are to be returned:
        # - Energy NRMSE.
        # - Time NRMSE.
        pass
    '''

    def computeEnergyTimeNormalizedErrors(self, params):
        '''
        Two float values are to be returned:
        - Energy NRMSE.
        - Time NRMSE.
        '''
        energyRMSE, timeRMSE = self.computeEnergyTimeErrors(params)
        realCosts = self.getRealEnergyTimeCosts()
        energyCosts = np.array(realCosts['energy']).flatten()
        timeCosts = np.array(realCosts['time']).flatten()

        stdEnergyCosts: float = energyCosts.std()
        stdTimeCosts: float = timeCosts.std()

        timeNRMSE = timeRMSE / stdTimeCosts
        energyNRMSE = energyRMSE / stdEnergyCosts
        return energyNRMSE, timeNRMSE

    def paramsListToDict(self, listParams):
        pass

    def paramsDictToList(self, params):
        pass









