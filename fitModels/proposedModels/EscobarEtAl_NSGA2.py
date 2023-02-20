import random
from fitModels.proposedModels.NSG2 import NSGA2

from energyTimeModels.proposedModels import EscobarEtAl
from fitModels.FitModel import FitModel

# Función que define valores uniformes entre dos cotas:
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]




defaultHyperparameters = {
    'numGenerations': 100,
    'numIndividuals': 120,
    'minBound': 0.0,
    'maxBound': 1.0,
    'childParentSimilarity': 1.0,
    'genMutationRate': 'uniform',
    'crossRate': 0.5,
}

class EscobarEtAl_NSG2(FitModel):
    data = dict()
    hyperparameters = dict()
    paramsMinMax = dict()

    def __init__(self, hyperparameters=dict(), data=dict(), paramsMinMax=dict()):
        super().__init__(hyperparameters, data, paramsMinMax)
        if hyperparameters == dict():
            self.hyperparameters = dict(defaultHyperparameters)

    def fit(self, model: EscobarEtAl):
        # Given an EnergyTimeModel, its parameters are fitted according to the
        # implementation of FitModel. A dictionary with a set of results
        # and metrics is returned:

        # First, we implement NSGA2 for time parameters fitting:
        genMutationRate = self.hyperparameters['genMutationRate']
        if genMutationRate == 'uniform':
            genMutationRate = 1.0 / len(model.hyperparameters['timeParams'])

        numTimeParams = len(model.hyperparameters['timeParams'])
        numEnergyParams = len(model.hyperparameters['energyParams'])

        fitnessFunction = lambda individual: [1.0, self.computeFitnessTime(model, individual, True)[1]]
        initIndividualFunction = lambda : [0.1] * numTimeParams
        #initIndividualFunction = lambda : uniform(self.hyperparameters['minBound'],
        #                                          self.hyperparameters['maxBound'],
        #                                          numTimeParams)
        timeFittingAlgorithm = NSGA2(
            numGens=numTimeParams,
            fitnessFunction=fitnessFunction,
            initIndividualFunction=initIndividualFunction,
            boundLow=self.hyperparameters['minBound'],
            boundUp=self.hyperparameters['maxBound'],
            numGenerations=self.hyperparameters['numGenerations'],
            numIndividuals=self.hyperparameters['numIndividuals'],
            crossoverRate=self.hyperparameters['crossRate'],
            genMutationRate=genMutationRate,
            childParentSimilarity=self.hyperparameters['childParentSimilarity']
        )
        resultingPop, stats = timeFittingAlgorithm.fit()
        resultingPop.sort(key=lambda x: x.fitness.values[1])
        bestIndividualTime = resultingPop[0]

        fitnessFunction = lambda individual: [self.computeFitness(model, individual+bestIndividualTime, True)[0], 1.0]
        initIndividualFunction = lambda : [0.1] * numEnergyParams
        energyFittingAlgorithm = NSGA2(
            numGens=numEnergyParams,
            fitnessFunction=fitnessFunction,
            initIndividualFunction=initIndividualFunction,
            boundLow=self.hyperparameters['minBound'],
            boundUp=self.hyperparameters['maxBound'],
            numGenerations=self.hyperparameters['numGenerations'],
            numIndividuals=self.hyperparameters['numIndividuals'],
            crossoverRate=self.hyperparameters['crossRate'],
            genMutationRate=genMutationRate,
            childParentSimilarity=self.hyperparameters['childParentSimilarity']
        )
        resultingPop, stats = energyFittingAlgorithm.fit()
        resultingPop.sort(key=lambda x: x.fitness.values[0])
        bestIndividualEnergy = resultingPop[0]

        bestIndividualEnergyTime = bestIndividualEnergy + bestIndividualTime
        results =  {
            'bestIndividual': model.denormalizeParams(model.paramsListToDict(bestIndividualEnergyTime, energyParameters=True,
                                                                   timeParameters=True)),
            'bestFitness': self.computeFitness(model, bestIndividualEnergyTime, areParamsNorm=True)
        }
        return results

    '''
    def computeFitness(self, model: EscobarEtAl, params, areParamsNorm = False):
        return model.computeEnergyTimeErrors(params, areParamsNorm)
    '''
    '''
    def computeFitnessTime(self, model: EscobarEtAl, params, areParamsNorm = False):
        
        params_ = params
        # If params is a list of time parameters, we transform it into a dictionary:
        if params is list:
            params_ = dict()
            for i in range(0, len(self.hyperparameters['energyParams'])):
                energyParam = self.hyperparameters['energyParams'][i]
                params_[energyParam] = 1e-9
            for i in range(0, len(self.hyperparameters['timeParams'])):
                timeParam = self.hyperparameters['timeParams'][i]
                params_[timeParam] = params[i]
        # return self.computeFitness(model, params_, areParamsNorm)
        return model.computeEnergyTimeErrors(params_, areParamsNorm)
    '''
    def computeFitnessTime(self, model: EscobarEtAl, params, areParamsNorm=False):

        params_ = params
        # If params is a list of time parameters, we transform it into a dictionary:
        if type(params) is not dict():
            params_ = dict()
            for i in range(0, len(model.hyperparameters['energyParams'])):
                energyParam = model.hyperparameters['energyParams'][i]
                params_[energyParam] = 1e-9
            params_.update(model.paramsListToDict(params, energyParameters=False, timeParameters=True))
        # return self.computeFitness(model, params_, areParamsNorm)
        return model.computeEnergyTimeErrors(params_, areParamsNorm)

    '''
    def computeFitness(self, model: EscobarEtAl, params, areParamsNorm = False):
        params_ = params
        # If params is a list of time parameters, we transform it into a dictionary:
        if params is list:
            params_ = dict()
            for i in range(0, len(model.hyperparameters['energyParams'])):
                energyParam = model.hyperparameters['energyParams'][i]
                params_[energyParam] = params[i]
            for i in range(0, len(model.hyperparameters['timeParams'])):
                timeParam = model.hyperparameters['timeParams'][i]
                params_[timeParam] = params[i
                    + len(model.hyperparameters['energyParams'])]

        # return self.computeFitness(model, params_, areParamsNorm)
        return model.computeEnergyTimeErrors(params_, areParamsNorm)
    '''

    def computeFitness(self, model: EscobarEtAl, params, areParamsNorm = False):
        params_ = params
        # If params is a list of time parameters, we transform it into a dictionary:
        if type(params) is not dict:
            params_ = model.paramsListToDict(params, energyParameters=True, timeParameters=True)

        # return self.computeFitness(model, params_, areParamsNorm)
        return model.computeEnergyTimeErrors(params_, areParamsNorm)

    '''
    def __initIndividual(self, model):
        for indice in range(1, 4):
            i = str(indice)
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
                parametros.append(datos['Wgpu' + i] / self.MAX_VALOR_W)
                parametros.append(datos['Wcpu' + i] / self.MAX_VALOR_W)
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
                parametros.append(min(datos['POW_gpu' + i] / self.MAX_VALOR_Ps[i + '_1'], 1.0))
                parametros.append(min(datos['POW_cpu' + i] / self.MAX_VALOR_Ps[i + '_2'], 1.0))
                # parametros.append(datos['POW_gpu' + i + '_idle'] / self.MAX_VALOR_Ps[i + '_1'])
                # parametros.append(datos['POW_cpu' + i + '_idle'] / self.MAX_VALOR_Ps[i + '_2'])

                parametros.append(min(datos['POW_gpu' + i + '_idle'] / datos['POW_gpu' + i], 1.0))
                parametros.append(min(datos['POW_cpu' + i + '_idle'] / datos['POW_cpu' + i], 1.0))

        # Otros parámetros a ajustar:
        if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
            parametros.append(datos['Tcom'] / self.MAX_VALOR_T)
            parametros.append(datos['Tmaster'] / self.MAX_VALOR_T)
        if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
            parametros.append(min(datos['POW_cpu0'] / self.MAX_VALOR_P_MASTER, 1.0))
            parametros.append(min(datos['POW_sw'] / self.MAX_VALOR_P_SW, 1.0))
    '''
