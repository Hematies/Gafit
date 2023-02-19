from fit.Fit import Fit
from energyTimeModels.proposedModels.EscobarEtAl import EscobarEtAl
from fitModels.proposedModels.EscobarEtAl_NSGA2 import EscobarEtAl_NSG2

def test():
    energyTimeModel = EscobarEtAl()
    fitModel = EscobarEtAl_NSG2()
    frontend = Fit(data=energyTimeModel.getData(), energyTimeModel=energyTimeModel, fitModel=fitModel)
    results = frontend.fit()
    e = energyTimeModel.computeEnergyTimeNormalizedErrors(results['bestIndividual'])

    print('')

if __name__ == '__main__':
    test()

