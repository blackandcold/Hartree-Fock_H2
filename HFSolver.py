import numpy as np
import math
from scipy import *
from scipy.linalg import eigh, eig


class HFSolver:
    dt = float

    def __init__(self, base, nucleiPositions, numberOfElectrons,  breakValue=1e-8, maximumIterations = 400):
        self.base = base
        self.numberOfElectrons = numberOfElectrons
        self.nucleiPositions = nucleiPositions
        self.breakValue = (breakValue)
        self.maximumIterations = maximumIterations

    def CalculateNuclearRepulsion(self, R_AB):
        return (1*1./R_AB)  # Z_n is 1, it only depends on R_AB

    def CalculateTotalEnergy(self, eigenValues, R_AB, rho, simpleHamiltonian):
        totalE = 0.

        totalE = sum(rho.transpose() * simpleHamiltonian)  # 1/2 * 2 * h_rs * P_rs

        for i in range(self.numberOfElectrons):
            totalE += 0.5 * eigenValues[i/2]  # i/2 is always 0 here for H2 since we only take half the electrons into account (Angabe!)
        totalE += self.CalculateNuclearRepulsion(R_AB)
        return totalE

    def SolveGeneralEigensystem(self, fockMatrix, overlapMatrix):
        eigenValues, eigenVectors = eigh(fockMatrix, overlapMatrix)  # generalized eigenvalue problem of symmetic matrices, "eig" only works for one input!
        eigenValues = eigenValues[:2]
        return (eigenValues, eigenVectors[:,:2])

    def CalcualteTwoEIntegral(self, overlapMatrix, R_AB):
        Qpqrs = np.zeros((len(self.base), len(self.base), len(self.base), len(self.base)), dtype=float)
        for p, alphaP in enumerate(self.base):
            for q, alphaQ in enumerate(self.base):
                for r, alphaR in enumerate(self.base):
                    for s, alphaS in enumerate(self.base):
                        R_pq = 0.5 * R_AB * (self.nucleiPositions[p] * alphaP + self.nucleiPositions[r] * alphaR) / (alphaP + alphaR)
                        R_rs = 0.5 * R_AB * ((self.nucleiPositions[q]) * alphaQ + (self.nucleiPositions[s]) * alphaS) / (alphaQ + alphaS)
                        alphaPQRS = (alphaP + alphaR) * (alphaQ + alphaS) / ((alphaP + alphaR) + (alphaQ + alphaS))
                        #  division by np.sqrt(np.pi) because twice the overlap is pi**6/2 but two electron only has pi**5/2
                        Qpqrs[p, q, r, s] = 2 / sqrt(np.pi) * sqrt(alphaPQRS) * overlapMatrix[p, r] * overlapMatrix[q, s] * self.F0(alphaPQRS * (R_pq - R_rs) ** 2)
        return Qpqrs

    def CalculateFockMatrix(self, simpleHamiltonian, Qpqrs, rho):
        fockMatrix = np.zeros((len(self.base), len(self.base)), dtype=self.dt)

        # Fpq = hpq + 1/2 sum(rho * Qpqrs)
        for p, alphaP in enumerate(self.base):
            for q, alphaQ in enumerate(self.base):
                sumRhoQpqrs = 0
                for r, alphaR in enumerate(self.base):
                    for s, alphaS in enumerate(self.base):
                        sumRhoQpqrs += rho[r, s] * (2 * Qpqrs[r, p, s, q] - Qpqrs[r, p, q, s])

                fockMatrix[p, q] = simpleHamiltonian[p, q] + sumRhoQpqrs

        return fockMatrix


    # From http://edu.itp.phys.ethz.ch/fs12/cqp/exercise07.pdf
    def F0(self, q):
        #return np.erf(x)
        if abs(q) < 1e-18:  # due to "ValueError: array must not contain infs or NaNs", getting so small that it is interpreted as 0
            return 1.0
        q = np.sqrt(q)
        return sqrt(np.pi)/2 * math.erf(q) / q

    def CalculateOverlapMatrix(self, R_AB):
        overlapMatrix = np.zeros((len(self.base), len(self.base)), dtype=self.dt)
        for p, alphaP in enumerate(self.base):
            for q, alphaQ in enumerate(self.base):
                overlapMatrix[p][q] = (pi/(alphaP+alphaQ))**(3/2.)
                if self.nucleiPositions[p] != self.nucleiPositions[q]:
                    overlapMatrix[p][q] *= exp(-R_AB**2 * (alphaP * alphaQ)/(alphaP + alphaQ))

        return overlapMatrix

    def CalculateSimpleHamiltonian(self, R_AB, overlapMatrix):
        simpleHamiltonian = np.zeros((len(self.base), len(self.base)), dtype=self.dt)
        for p, alphaP in enumerate(self.base):
            for q, alphaQ in enumerate(self.base):
                alphaPQ = (alphaP * alphaQ) / (alphaP + alphaQ)
                R_pq = 0.5 * R_AB * (self.nucleiPositions[p] * alphaP + self.nucleiPositions[q] * alphaQ) / (alphaP + alphaQ)

                # Kinetic term part
                simpleHamiltonian[p][q] = 3 * overlapMatrix[p][q] * alphaPQ
                if self.nucleiPositions[p] != self.nucleiPositions[q]:
                    simpleHamiltonian[p][q] *= (1 - 2/3. *alphaPQ * R_AB**2)

                # Nuclear attraction part
                # the two nuclei are located at |R_p + R_C| and |R_p - R_C| => center of orbital is in general not where the nuclei are
                F0_plus = self.F0((alphaP+alphaQ) * (R_pq + R_AB/2.)**2)
                F0_minus = self.F0((alphaP+alphaQ) * (R_pq - R_AB/2.)**2)

                # np.sqrt((alphaP + alphaQ) / np.pi) is to compensate additional overlap factor
                simpleHamiltonian[p][q] += -2*overlapMatrix[p][q] * np.sqrt((alphaP+alphaQ)/np.pi) * (F0_plus + F0_minus)

        return simpleHamiltonian

    def SCF(self, rho, R_AB):
        overlapMatrix = self.CalculateOverlapMatrix(R_AB)
        simpleHamiltonian = self.CalculateSimpleHamiltonian(R_AB, overlapMatrix)
        Qpqrs = self.CalcualteTwoEIntegral(overlapMatrix, R_AB)

        # init current energy with 0
        currentE = 0
        for i in range(self.maximumIterations):
            fockMatrix = self.CalculateFockMatrix(simpleHamiltonian, Qpqrs, rho)
            (eigenValues, eigenVectors) = self.SolveGeneralEigensystem(fockMatrix, overlapMatrix)
            newRho = self.CalculateDensity(eigenVectors)
            newE = self.CalculateTotalEnergy(eigenValues, R_AB, newRho, simpleHamiltonian)

            if np.abs(currentE - newE) < self.breakValue:
                break
            rho = newRho
            currentE = newE

        return currentE, rho

    def CalculateDensity(self, eigenVectors):
        newRho = np.zeros((len(eigenVectors), len(eigenVectors)), dtype=self.dt)

        for i in range(self.numberOfElectrons):
            for j in range(len(eigenVectors)):
                for k in range(len(eigenVectors)):
                    # since they are real, order does not matter
                    newRho[j][k] += 0.5 * eigenVectors[j][i/2] * eigenVectors[k][i/2]

        return newRho




    #     self.R = Rpos # Positions of Rn
    #     self.distanceAB = R0
    #
    #     self.rpqMatrix = [] # "Center of mass" for nuclei wave funktions
    #     self.spqMatrix = [] # Overlap matrix base functions
    #     self.tpqMatrix = [] # Kinetic matrix elements
    #     self.vpqMatrix = [] # Nuclear attraction
    #     self.qprqsMatrix = [] #Kinetic and Coulomb
    #     self.ppqMatrix = [] #Density
    #     self.fpqMatrix = [] #Fock Matrix
    #     self.totalE = (0)
    #     self.z = (1) # Atomic number
    #     self.totHamiltonian = []
    #     self.pCoeff = [(0.3), (0.01), (0.7), (0.9), (0.3), (0.01), (0.7), (0.9)]
    #     self.pCoeffMatrix = np.outer(self.pCoeff, self.pCoeff)
    #
    #     # Initialise matrices without C dependence
    #     self.Rpq()
    #     self.Spq()
    #     self.Tpq()
    #     self.Vpq()
    #     self.Qprqs()
    #     self.guessPpq()
    #     self.Fpq()
    #
    #
    # def Rpq(self):
    #     self.rpqMatrix = np.zeros((len(self.base), len(self.base)), dtype=self.dt)
    #     for i, alphaI in enumerate(self.base):
    #         for j, alphaJ in enumerate(self.base):
    #             value = (alphaI * self.R[i] + alphaJ * self.R[j]) / (np.add(alphaI, alphaJ))
    #             self.rpqMatrix[i][j] = value
    #
    # def Spq(self):
    #     self.spqMatrix = np.zeros((len(self.base), len(self.base)), dtype=self.dt)
    #     for i, alphaI in enumerate(self.base):
    #         for j, alphaJ in enumerate(self.base):
    #             value = np.power((np.pi / (np.add(alphaI, alphaJ))), 3/2.)
    #             if self.R[i] != self.R[j]: # exp vanishes if they are the same
    #                 value *= np.exp(-((alphaI*alphaJ) / (np.add(alphaI, alphaJ))) * np.power(np.subtract(2*self.R[i]-1, 2*self.R[j]-1), 2))  # replacing Kpq
    #             self.spqMatrix[i][j] = value
    #
    # def Tpq(self):
    #     self.tpqMatrix = np.zeros((len(self.base), len(self.base)), dtype=self.dt)
    #     for i, alphaI in enumerate(self.base):
    #         for j, alphaJ in enumerate(self.base):
    #             value = ((alphaI*alphaJ) / (np.add(alphaI, alphaJ))) * (6-4*((alphaI*alphaJ) / (np.add(alphaI, alphaJ))) * np.subtract(self.R[i], self.R[j])) * self.spqMatrix[i][j]
    #             self.tpqMatrix[i][j] = value
    #
    # def Vpq(self):
    #     self.vpqMatrix = np.zeros((len(self.base), len(self.base)), dtype=self.dt)
    #     for i, alphaI in enumerate(self.base):
    #         for j, alphaJ in enumerate(self.base):
    #             diff = np.abs(self.rpqMatrix[i][j]-self.distanceAB)
    #             if diff == 0:
    #                 value = -2 * np.pi / np.add(alphaJ, alphaI)
    #             else:
    #                 value = - self.spqMatrix[i][j] * (1/diff) * math.erf(math.sqrt(np.add(alphaI, alphaJ)) * abs(diff))
    #             self.vpqMatrix[i][j] = value
    #
    # def Qprqs(self):
    #     self.qprqsMatrix = np.zeros((len(self.base), len(self.base), len(self.base), len(self.base)), dtype=self.dt)
    #     for p, alphaP in enumerate(self.base):
    #         for r, alphaR in enumerate(self.base):
    #             for q, alphaQ in enumerate(self.base):
    #                 for s, alphaS in enumerate(self.base):
    #                     absDiff = abs(self.rpqMatrix[p][q] - self.rpqMatrix[r][s])
    #                     if absDiff != 0:
    #                         alphaFactor = math.sqrt(((np.add(alphaP, alphaQ)) * np.add(alphaR, alphaS)) /
    #                                             np.add(np.add(alphaP, alphaQ), np.add(alphaR, alphaS)))
    #                         self.qprqsMatrix[p][r][q][s] = self.spqMatrix[p][q] * self.spqMatrix[r][s] * 1/absDiff * math.erf(alphaFactor * absDiff)
    #
    # def guessPpq(self):
    #     self.ppqMatrix = np.zeros((len(self.base), len(self.base)), self.dt)
    #     for p, alphaP in enumerate(self.base):
    #         for q, alphaQ in enumerate(self.base):
    #             ckSum = 0
    #             for k, alphaK in enumerate(self.base):
    #                 ckSum += self.pCoeffMatrix[p][k] * self.pCoeffMatrix[q][k]
    #             self.ppqMatrix[p][q] = 2 * ckSum
    #
    # def calculateEVProblem(self):
    #     eigvals, eigvecs = eig(self.fpqMatrix, self.spqMatrix)
    #     print eigvals
    #     print eigvecs
    #     #eigvals, eigvecs = eigh(self.fpqMatrix, self.spqMatrix, type=1)
    #     #print eigvals
    #     #print eigvecs
    #
    #
    # def Fpq(self):
    #     sumOfContributions = np.zeros((len(self.base), len(self.base)), dtype=self.dt)
    #     for p, alphaP in enumerate(self.base):
    #             for q, alphaQ in enumerate(self.base):
    #                 for r, alphaR in enumerate(self.base):
    #                     for s, alphaS in enumerate(self.base):
    #                         sumOfContributions[r][s] = self.ppqMatrix[s][r] * (2 * self.qprqsMatrix[p][r][q][s] - self.qprqsMatrix[p][r][s][q])
    #
    #                 self.fpqMatrix = self.tpqMatrix + (0.5) * sumOfContributions
    #
    # def totalEnergy(self):
    #     Etot = (0)
    #     for p, alphaP in enumerate(self.base):
    #         for q, alphaQ in enumerate(self.base):
    #             cksum = (0)
    #             for k, alphaK in enumerate(self.base):
    #                 ckSum += self.pCoeffMatrix[p][k] * self.pCoeffMatrix[q][k]
    #             self.ppqMatrix[p][q] = 2 * ckSum
