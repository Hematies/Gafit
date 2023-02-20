from fit.Fit import Fit
from energyTimeModels.proposedModels.EscobarEtAl import EscobarEtAl
from fitModels.proposedModels.EscobarEtAl_NSGA2 import EscobarEtAl_NSG2
import json

def test():
    energyTimeModel = EscobarEtAl()
    fitModel = EscobarEtAl_NSG2()
    frontend = Fit(data=energyTimeModel.getData(), energyTimeModel=energyTimeModel, fitModel=fitModel)
    results = frontend.fit()
    e = energyTimeModel.computeEnergyTimeNormalizedErrors(results['bestIndividual'])

    results['NRMSE_E'] = e[0]
    results['NRMSE_T'] = e[1]
    with open("test.json", "w") as outfile:
        json.dump(results, outfile)

    print('')

if __name__ == '__main__':
    test()

