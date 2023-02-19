import math

import scipy as scipy

from energyTimeModels.EnergyTimeModel import EnergyTimeModel
import scipy.io as scipy

defaultHyperparameters = {
    'TDP_margin': 1.5,
    'min_pow': 1,
    'max_w': 1e9,
    'max_t': 5.0,
    'maxNumSubp': 32,
    'isTimeInMili': True,
    'nodesDevicesMatrix': [2, 2, 2], # This should be in self.data
    'crossRate': 0.75, # This should be in self.data
    'energyParams': [
        'Pow_1_1',
        'Pow_1_2',
        'Pow_2_1',
        'Pow_2_2',
        'Pow_3_1',
        'Pow_3_2',
        'POW_sw',
        'POW_cpu0',
        'Pow_1_1_idle',
        'Pow_1_2_idle',
        'Pow_2_1_idle',
        'Pow_2_2_idle',
        'Pow_3_1_idle',
        'Pow_3_2_idle',
        'POW_sw_idle',
        'POW_cpu0_idle',
    ],
    'timeParams': [
        'W_1_1',
        'W_1_2',
        'W_2_1',
        'W_2_2',
        'W_3_1',
        'W_3_2',
        'T_com',
        'T_master'
    ],

}

powMargin = defaultHyperparameters['TDP_margin']
minPow = defaultHyperparameters['min_pow']
maxW = defaultHyperparameters['max_w']
maxT = defaultHyperparameters['max_t']

defaultParamsMinMax = {
    'W_1_1': (10e-9,maxW),
    'W_1_2': (10e-9,maxW),
    'W_2_1': (10e-9,maxW),
    'W_2_2': (10e-9,maxW),
    'W_3_1': (10e-9,maxW),
    'W_3_2': (10e-9,maxW),
    'T_com': (10e-9,maxT),
    'T_master': (10e-9,maxT),

    'Pow_1_1': (minPow, 225 * powMargin),
    'Pow_1_2': (minPow, 80 * powMargin),
    'Pow_2_1': (minPow, 250 * powMargin),
    'Pow_2_2': (minPow, 85 * powMargin),
    'Pow_3_1': (minPow, 250 * powMargin),
    'Pow_3_2': (minPow, 85 * powMargin),
    'POW_sw': (minPow, 5 * powMargin),
    'POW_cpu0': (minPow, 80 * powMargin),

    'Pow_1_1_idle': (minPow, 'Pow_1_1'),
    'Pow_1_2_idle': (minPow, 'Pow_1_2'),
    'Pow_2_1_idle': (minPow, 'Pow_2_1'),
    'Pow_2_2_idle': (minPow, 'Pow_2_2'),
    'Pow_3_1_idle': (minPow, 'Pow_3_1'),
    'Pow_3_2_idle': (minPow, 'Pow_3_2'),
    'POW_sw_idle': (minPow, 'POW_sw'),
    'POW_cpu0_idle': (minPow, 'POW_cpu0'),
}

class EscobarEtAl(EnergyTimeModel):

    def __init__(self, hyperparameters = dict(), data = dict(), paramsMinMax = dict(),
                 dataFile = "./data/workspc_30N.mat"):
        super().__init__(hyperparameters, data, paramsMinMax)
        if hyperparameters == dict():
            self.hyperparameters = dict(defaultHyperparameters)
        if paramsMinMax == dict():
            self.paramsMinMax = dict(defaultParamsMinMax)
        if data == dict():
            # We import the .mat workspace into the model:
            self.dataFile = dataFile
            self.data = scipy.loadmat(dataFile)
            self.__translateWorkspaceParams()


    def computeEnergyTimeErrors(self, params, areParamsNorm = False):
        return self.__computeModel(params, areParamsNorm,
           returnCosts=False)

    def __computeModel(self, params, areParamsNorm = False,
        returnCosts = False):
        if areParamsNorm:
            return self.__computeModel(self.denormalizeParams(params), False)
        else:

            aux = self.getRealEnergyTimeCosts()
            realTimes = aux['time']
            realEnergies = aux['energy']

            timeRMSE = 0.0
            energyRMSE = 0.0

            # We compute costs:
            computedCosts = self.computeEnergyTimeCosts(params)

            realCosts = {
                'time': realTimes,
                'energy': realEnergies
            }

            for numSubpop in range(1, self.hyperparameters['maxNumSubp'] + 1):
                realTime = realTimes[numSubpop - 1]
                realEnergy = realEnergies[numSubpop - 1]
                computedTime = computedCosts['time'][numSubpop - 1]
                computedEnergy = computedCosts['energy'][numSubpop - 1]

                timeRMSE = float(timeRMSE + \
                                 (1 / self.hyperparameters['maxNumSubp'] * \
                                  math.pow(computedTime - realTime, 2)))
                energyRMSE = float(energyRMSE + \
                                   (1 / self.hyperparameters['maxNumSubp'] * \
                                    math.pow(computedEnergy - realEnergy, 2)))

            timeRMSE = math.sqrt(timeRMSE)
            energyRMSE = math.sqrt(energyRMSE)


            if not returnCosts:
                return energyRMSE, timeRMSE
            else:
                return realCosts, computedCosts

    def getRealEnergyTimeCosts(self):
        realTimes = self.data["time_v1v2v3_all"][:, 2]
        if self.hyperparameters['isTimeInMili']:
            realTimes = list(map(lambda t: float(t) / 1000, realTimes))
        realEnergies = self.data["energ_v1v2v3_all"][:, 2]
        return {
            'time': realTimes,
            'energy': realEnergies
        }

    def computeEnergyTimeCosts(self, params):
        res = {'time': [], 'energy': []}
        res_ = []
        for numSubpop in range(1, self.hyperparameters['maxNumSubp'] + 1):
            res_.append(self.__computeEnergyTimeCosts(params, numSubpop))
        for numSubpop in range(1, self.hyperparameters['maxNumSubp'] + 1):
            res['time'].append(res_[numSubpop-1]['time'])
            res['energy'].append((res_[numSubpop-1]['energy']))
        return res

    def __computeEnergyTimeCosts(self, params, numTotalSubpop):
        # First, we compute time costs.
        # For such purpose, the workload distribution is to be estimated
        # based on given time parameters. Additionally, total cluster runtime
        # between migrations is returned:
        workloadMatrix, totalTimeBetMig, activeCommTimeBetMig = self.__simulateWorkloadDistr(params, numTotalSubpop)

        # Then, we can compute total cluster time cost for all migrations :
        totalTime = float(self.data['NGmig'] * totalTimeBetMig)

        # Having calculated time costs, device energy costs between migrations are estimated:
        totalEnergyBetMig = 0.0
        clusterMatrix = self.hyperparameters['nodesDevicesMatrix']
        for nodeIndex in range(0, len(clusterMatrix)):
            for devIndex in range(0, clusterMatrix[nodeIndex]-1):
                energy = self.__computeDevEnergyCost(params, numTotalSubpop,
                                                     workloadMatrix, totalTimeBetMig,
                                                     nodeIndex + 1, devIndex + 1)
                totalEnergyBetMig = totalEnergyBetMig + energy

        # Then, communications and master node costs:
        totalEnergyBetMig = totalEnergyBetMig + self.__computeCommCost(params, activeCommTimeBetMig, totalTimeBetMig)
        totalEnergyBetMig = totalEnergyBetMig + self.__computeMasterCost(params, totalTimeBetMig)

        # Finally, we compute total cluster energy consumption and return the results:
        totalEnergy = float(self.data['NGmig'] * totalEnergyBetMig) / 3600

        return {'energy': totalEnergy, 'time': totalTime}


    def __simulateWorkloadDistr(self, params, numTotalSubpop):
        workload = []

        # Phase = Timestamp in which there are one or more subpopulations to be
        # computed for one or more devices in state 'ready'.
        numDevs = 0
        subpopDevTimeCosts = []
        nodeDevRanking = []

        # The data structures which make use of scheduling are initialized:
        clusterMatrix = self.hyperparameters['nodesDevicesMatrix']
        for nodeIndex in range(0, len(clusterMatrix)):
            numDevs = numDevs + clusterMatrix[nodeIndex]
            workload.append([])
            subpopDevTimeCosts.append([])

            # We compute active time cost of one subpopulation for each device:
            for devIndex in range(0, clusterMatrix[nodeIndex]):
                workload[nodeIndex].append(0)

                # We skip the device if all cores are disabled:
                if self.data['P_'+str(nodeIndex+1)+'_'+str(devIndex+1)] > 0:
                    subpopDevTimeCosts[nodeIndex].append(
                        self.__computeDevTimeCost(params, numTotalSubpop, 1,
                                                  nodeIndex + 1, devIndex + 1)
                    )
                    nodeDevRanking.append({
                        'nodeIndex': nodeIndex,
                        'devIndex': devIndex,
                        'readyTimestamp': 0.0,
                    })
                else:
                    subpopDevTimeCosts[nodeIndex].append(0)

        # Subpopulations are delivered until none remains:
        numLastingSubpop = numTotalSubpop
        commPeriods = TimePeriods()
        while numLastingSubpop > 0:
            # Next device to which a subpopulation is assigned is chosen as the next
            # device to enter 'ready' state:
            nodeIndex = nodeDevRanking[0]['nodeIndex']
            devIndex = nodeDevRanking[0]['devIndex']
            timestamp = nodeDevRanking[0]['readyTimestamp']

            # Device ready timestamp is updated, plus communications cost:
            newTimestamp = timestamp + subpopDevTimeCosts[nodeIndex][devIndex] + params['T_com']
            nodeDevRanking[0]['readyTimestamp'] = newTimestamp

            commPeriods.addPeriod(timestamp + subpopDevTimeCosts[nodeIndex][devIndex], newTimestamp)

            # We update the resulting workload:
            workload[nodeIndex][devIndex] = workload[nodeIndex][devIndex] + 1
            numLastingSubpop = numLastingSubpop - 1

            # Finally, we reorder the ranking for having next available device at the head:
            nodeDevRanking.sort(key=lambda dev: dev['readyTimestamp'])

        # Total cluster time for computing a full evolution between two migrations is returned:
        totalTime = max(nodeDevRanking, key=lambda dev: dev['readyTimestamp'])['readyTimestamp']
        totalTime = totalTime + params['T_master'] # Overhead time costs are added.

        # Total active communication time is computed by summing the duration of the comm periods:
        activeCommTime = commPeriods.getTotalTime()

        return workload, totalTime, activeCommTime

    def __computeDevTimeCost(self, params, numTotalSubpop, numSubpop,
        nodeIndex, devIndex):
        i = str(nodeIndex)
        j = str(devIndex)

        numTotalInd = self.data['N']
        crossRate = self.hyperparameters['crossRate']
        numGenerations = self.data['gen'] / self.data['NGmig']
        numActCores = self.data['P_'+i+'_'+j]
        numClockCycles = params['W_'+i+'_'+j]
        frequency = self.data['F_'+i+'_'+j]

        numIndSubp = (numTotalInd / numTotalSubpop) * (1 + crossRate)
        if numActCores > 0:
            runtime = numGenerations * numSubpop * \
                      (math.ceil(numIndSubp / numActCores)) * \
                      numClockCycles / frequency
        else:
            runtime = 0
        return runtime

    def __computeDevEnergyCost(self, params, numTotalSubpop, workloadMatrix, totalTime,
                             nodeIndex, devIndex):
        i = str(nodeIndex)
        j = str(devIndex)
        numSubpop = workloadMatrix[nodeIndex-1][devIndex-1]
        activeTime = self.__computeDevTimeCost(params, numTotalSubpop, numSubpop,
                                               nodeIndex, devIndex)
        idleTime = totalTime - activeTime
        assert idleTime >= 0

        return params['Pow_'+i+'_'+j] * activeTime + params['Pow_'+i+'_'+j+'_idle'] * idleTime

    def __computeMasterCost(self, params, totalTime):
        activeTime = params['T_master']
        idleTime = totalTime - activeTime
        assert idleTime >= 0
        return params['POW_cpu0'] * activeTime + params['POW_cpu0_idle'] * idleTime

    def __computeCommCost(self, params, activeCommTime, totalTime):
        activeTime = activeCommTime
        idleTime = totalTime - activeTime
        assert idleTime >= 0
        return params['POW_sw'] * activeTime + params['POW_sw_idle'] * idleTime


    def paramsListToDict(self, listParams, energyParameters=True, timeParameters=True):
        params_ = dict()
        if energyParameters:
            for i in range(0, len(self.hyperparameters['energyParams'])):
                energyParam = self.hyperparameters['energyParams'][i]
                params_[energyParam] = listParams[i]
        n = len(params_)
        if timeParameters:
            for i in range(0, len(self.hyperparameters['timeParams'])):
                timeParam = self.hyperparameters['timeParams'][i]
                params_[timeParam] = listParams[n + i]
        return params_

    def paramsDictToList(self, params, energyParameters=True, timeParameters=True):
        listParams = []
        if energyParameters:
            for i in range(0, len(self.hyperparameters['energyParams'])):
                energyParam = self.hyperparameters['energyParams'][i]
                listParams.append(params[energyParam])
        if timeParameters:
            for i in range(0, len(self.hyperparameters['timeParams'])):
                timeParam = self.hyperparameters['timeParams'][i]
                listParams.append(params[timeParam])
        return listParams



    # If the workspace used in Escobar et al is given, certain data constants are to be translated
    # with this method
    def __translateWorkspaceParams(self):
        clusterMatrix = self.hyperparameters['nodesDevicesMatrix']
        numNodes = len(clusterMatrix)
        for i in range(1, numNodes+1):
            numDevs = clusterMatrix[i-1]
            for j in range(1, numDevs+1):
                if 'Pgpu' + str(i) in self.data or 'Pcpu' + str(i) in self.data:
                    self.data['P_' + str(i) + '_' + str(j)] = self.data['Pgpu' + str(i)] if j == 1 \
                        else self.data['Pcpu' + str(i)]
                if 'Fgpu' + str(i) in self.data or 'Fcpu' + str(i) in self.data:
                    self.data['F_' + str(i) + '_' + str(j)] = self.data['Fgpu' + str(i)] if j == 1 \
                        else self.data['Fcpu' + str(i)]

class TimePeriods:
    def __init__(self):
        self.periods = []

    def addPeriod(self, start, end):
        assert start < end

        self.periods.sort(key=lambda p: p[0])

        # First, we check the connections with the rest of periods
        periodConnections = []
        for i in range(0, len(self.periods)):
            t0, t1 = self.periods[i]
            if t1 < start or t0 > end:
                pass
            else:
                # There is a connection with the period:
                periodConnections.append(i)

        # If there are not any connections, we simply add the new period:
        if len(periodConnections) <= 0:
            self.periods.append((start, end))
        else:
            # We merge all connected periods into one:
            start_ = start
            end_ = end
            for i in periodConnections:
                t0, t1 = self.periods[i]
                if t0 < start_:
                    start_ = t0
                if t1 > end_:
                    end_ = t1

            self.periods[periodConnections[0]] = (start_, end_)

            if len(periodConnections[1:]) > 0:
                indices = list(filter(lambda k: not k in periodConnections[1:], range(0, len(self.periods))))
                self.periods = list(map(lambda k: self.periods[k], indices))

    def getTotalTime(self):
        res = 0.0
        for t0, t1 in self.periods:
            res = res + (t1 - t0)
        return res
