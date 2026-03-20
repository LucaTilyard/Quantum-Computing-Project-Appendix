
# --- Dependencies ---
#Basic Tools
import time
import numpy as np
import matplotlib.pyplot as plt

#Circuit Construction
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

#Quasi-Classical Simulation 
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel





# --- Define Hamiltonian ---
def defineH(L,Jz):
    terms = []
    coeffs = []
    for i in range(L-1):
        terms.extend([
            "I"*i+"XX"+"I"*(L-i-2),
            "I"*i+"YY"+"I"*(L-i-2),
            "I"*i+"ZZ"+"I"*(L-i-2)
        ])
        coeffs.extend([-1,-1,-Jz])
    return SparsePauliOp(terms, coeffs)

# --- Define Observable ---
def defineOps(L):
    ops = []
    for i in range(L):
        ops.extend([
            SparsePauliOp(Pauli("I"*i+"X"+"I"*(L-i-1))),
            SparsePauliOp(Pauli("I"*i+"Y"+"I"*(L-i-1))),
            SparsePauliOp(Pauli("I"*i+"Z"+"I"*(L-i-1)))
        ])
    return ops

# --- Estimator Setup ---
def setAerEstimator(L,noisy,dd,zne):
    if not noisy:
        estimator = StatevectorEstimator()
        pm = None
    else:
        fake_backend = GenericBackendV2(L)
        noise_model = NoiseModel.from_backend(fake_backend)
        aer_sim = AerSimulator(noise_model=noise_model)
        estimator = IBMEstimator(mode=aer_sim)
        pm = generate_preset_pass_manager(target=aer_sim.target, optimization_level=3)
    if dd:
        estimator.options.default_shots = 100000
        estimator.options.resilience_level = 0      
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XpXm"
    if zne:
        estimator.options.default_shots = 100000
        estimator.options.resilience.zne_mitigation = True
        estimator.options.resilience.zne.noise_factors = (1, 3, 5)
        estimator.options.resilience.zne.extrapolator = ("linear")
    return estimator,pm





# --- Run Time Evolution ---
def setInitialState(L,Jz):
    qc_init = QuantumCircuit(L)
    if Jz<1:
        for i in range(L):
            if i%2 == 0:
                qc_init.x(i) 
    qc_init.ry(np.pi/4, L//2)  #Initial State Setup
    return qc_init

def blendFunction(float,blendType,blendPara):
    match blendType:
        case "linear":
            return float
        case "log":
            return np.log(1+(blendPara-1)*float)/np.log(blendPara)
        case "power":
            return float**blendPara

def trotterization_blend(qc_init,hamiltonian,ops,maxTrotterDepth,total_time,max_turns,blendType,blendPara,estimator,pm):
    times = []
    expectations = []
    L = qc_init.num_qubits
    for i in range(0,max_turns+1):
        # Monitoring the time usage
        if i%10 == 0:
            if i == 0:
                startTime = time.time()
            else:
                print(f"{i} Turns Completed.(Time Usage={time.time()-startTime}s)")
                startTime = time.time()

        qc = QuantumCircuit(L)
        qc.compose(qc_init, inplace=True)
        current_time = i*total_time/max_turns
        if i > 0:
            trotterDepth = np.ceil(blendFunction(i/max_turns,blendType,blendPara)*maxTrotterDepth)
            dt = current_time/trotterDepth
            gate = PauliEvolutionGate(hamiltonian*dt)
            circ = SuzukiTrotter(order=1, reps=1).synthesize(gate)
            for _ in range(int(trotterDepth)):
                qc.append(circ, range(L))
        if pm == None:
            result = estimator.run(pubs=[(qc,ops)]).result()
        else:
            result = estimator.run(pubs=[(pm.run(qc),ops)]).result()
        expectations.append((result[0].data.evs).reshape(L,3))
        times.append(current_time)
    return np.array(expectations), np.array(times)





# --- Exploration of Error Cause By Trotter (No Noise)
def exploreTrotterError(qc_init,hamiltonian,ops,maxTrotterDepth,total_time,estimator,pm):
    # Comparing the error cause by trotter and noise over different depth.
    depth = []
    expectations = []
    L = qc_init.num_qubits
    for i in range(1,maxTrotterDepth+1):
        qc = QuantumCircuit(L)
        qc.compose(qc_init, inplace=True)
        dt = total_time/i
        gate = PauliEvolutionGate(hamiltonian*dt)
        circ = SuzukiTrotter(order=1, reps=1).synthesize(gate)
        for _ in range(i):
            qc.append(circ, range(L))
        if pm == None:
            result = estimator.run(pubs=[(qc,ops)]).result()
        else:
            result = estimator.run(pubs=[(pm.run(qc),ops)]).result()
        expectations.append((result[0].data.evs)[0])
        depth.append(i)
    return np.array(expectations),np.array(depth)





# --- Finding the Optimized Depth ---
def cleanInvalidPoints(expectations):
    # Used this function to remove invalid low depths that accidentally closest to ideal value.
    invalidIndex = [0]
    for i in range(1,len(expectations)-1):
        deltaLeft = abs(expectations[i]-expectations[i-1])
        deltaRight = abs(expectations[i+1]-expectations[i])
        if  (deltaLeft+deltaRight)>0.15:
            # The absolute value used in test is get from noisy result of L=3, J_z=-5, qubit 1, Z spin.
            invalidIndex.append(i) 
    for index in invalidIndex:
        # A random large number that can definitely be ignored.
        expectations[index] = 100 
    return expectations, invalidIndex

def exploreOptimizedDepth(qc_init,hamiltonian,ops,maxTrotterDepth,total_time):
    # This will try each depth (until maximum depth) of a specific total time (with noise).
    # It output a optimized depth which is closest to ideal value (with the maximum depth).
    depth = []
    expectationsSingle = []
    L = qc_init.num_qubits
    startTime = time.time()

    qc = QuantumCircuit(L)
    qc.compose(qc_init, inplace=True)
    dt = total_time/maxTrotterDepth
    gate = PauliEvolutionGate(hamiltonian*dt)
    circ = SuzukiTrotter(order=1, reps=1).synthesize(gate)
    for _ in range(maxTrotterDepth):
        qc.append(circ, range(L))
    estimator,pm = setAerEstimator(L,noisy=False,dd=False,zne=False)
    result = estimator.run(pubs=[(qc,ops)]).result()
    properExpectationSingle = result[0].data.evs
    estimator,pm = setAerEstimator(L,noisy=True,dd=False,zne=False)

    for i in range(1,maxTrotterDepth+1):
        qc = QuantumCircuit(L)
        qc.compose(qc_init, inplace=True)
        dt = total_time/i
        gate = PauliEvolutionGate(hamiltonian*dt)
        circ = SuzukiTrotter(order=1, reps=1).synthesize(gate)
        for _ in range(i):
            qc.append(circ, range(L))
        result = estimator.run(pubs=[(pm.run(qc),ops)]).result()
        depth.append(i)
        expectationsSingle.append((result[0].data.evs)[0])

    cleanedExpectations, invalidIndex = cleanInvalidPoints(expectationsSingle)
    distancesToProper = np.abs(cleanedExpectations-properExpectationSingle)
    optimizedDepth = np.argmin(distancesToProper)+1
    return optimizedDepth, np.array(expectationsSingle)


def exploreForEachTimeStep(qc_init,hamiltonian,ops,maxTrotterDepth,max_turns,trails,total_time):
    """
    This will try each depth (until maximum trotter depth) with certain number of trails to take average result.
    The total time will be splited to a certain number (max_turns) of time steps to explore them each. 
    Only accept 1 qubit and 1 direction observable.
    """
    times = []
    optimizedDepthArray = []
    for i in range(1,max_turns+1):
        # Monitoring the time usage
        startTime = time.time()
        currentTime = i*total_time/max_turns
        optimizedDepthRepeat = []
        for _ in range(0,trails):
            optimizedDepth, expectations = exploreOptimizedDepth(qc_init,hamiltonian,ops,maxTrotterDepth,currentTime)
            optimizedDepthRepeat.append(optimizedDepth)
        optimizedDepthArray.append(np.mean(np.array(optimizedDepthRepeat)))
        times.append(currentTime)
        print(f"Total Evolution Time = {currentTime}s Optimize Completed. Trials = {trails}. OptimizedDepth = {np.mean(optimizedDepthRepeat)}(Time Usage={time.time()-startTime}s)")
    return np.array(optimizedDepthArray), np.array(times)


# --- Plot Diagram ---
# Trotter Error over Depth
def plotZFixedTime(depth,ideal,noisy,label):
    plt.clf
    plt.figure(figsize=(10,6))
    plt.plot(depth, ideal, label="ideal")
    plt.plot(depth, noisy, label="noisy")
    #plt.ylim(-2,2)
    plt.xlabel("Trotter Depth")
    plt.ylabel(f"Expectation")
    plt.title(f"Fixed Total Time of 1D Chain Qubit 1 Expectation\n (L={L},J_z={Jz},{label})")
    plt.legend()
    plt.grid(True)
    figName = f"Fixed_Time_Plot_{label}.png"
    plt.savefig(figName)

# Optimized Depth
def plotOptimizedDepth(times,optimizedDepthArray,label):
    plt.clf
    plt.figure(figsize=(10,6))
    plt.plot(times, optimizedDepthArray)
    plt.xlabel("Time(s)")
    plt.ylabel(f"Depth")
    plt.title(f"Optimized Depth Versus Time\n (L={L},J_z={Jz},{label})")
    plt.grid(True)
    figName = f"{label}_Plot_Optimize.png"
    plt.savefig(figName)





# --- Instances ---
# Hamiltonian Config 
L = 3
Jz = -5

# Initial State
qc_init = setInitialState(L,Jz)

# Trotter Error Analysis
def DoTrotterError():
    maxTrotterDepth = 50
    total_time = 1
    qubit = 0 # Start from 0
    opsPauliStr = "I"*qubit+"Z"+"I"*(L-qubit-1)
    ops = [SparsePauliOp(Pauli(opsPauliStr))]
    estimator,pm = setAerEstimator(L,noisy=False,dd=False,zne=False)
    Ideal,depth = exploreTrotterError(qc_init,defineH(L,Jz),ops,maxTrotterDepth,total_time,estimator,pm)
    estimator,pm = setAerEstimator(L,noisy=True,dd=False,zne=False)
    Noisy,depth = exploreTrotterError(qc_init,defineH(L,Jz),ops,maxTrotterDepth,total_time,estimator,pm)
    plotZFixedTime(depth,Ideal,Noisy,f"{total_time}_{maxTrotterDepth}")

def DoDepthOptimize():
    maxTrotterDepth = 20
    max_turns = 100
    trails = 1
    total_time = 1
    qubit = 0 # Start from 0
    opsPauliStr = "I"*qubit+"X"+"I"*(L-qubit-1)
    ops = [SparsePauliOp(Pauli(opsPauliStr))]
    optimizedDepthArray, times = exploreForEachTimeStep(qc_init,defineH(L,Jz),ops,maxTrotterDepth,max_turns,trails,total_time)
    fileName = f"Optimized_Depth_mt{max_turns}_tt{total_time}_tr{trails}_{opsPauliStr}.npy"
    np.save(fileName,np.array([optimizedDepthArray,times])) # Save result for possible further analysis.
    plotOptimizedDepth(times, optimizedDepthArray, f"TotalTime_{total_time}_{opsPauliStr}")
    return fileName

def AdaptingAnalysis(fileName):
    """
    Since the result plot shows a periodic behavier (Because of the inital low depths have sharp error fluctuations), 
    its hard to analyse the shape of the curve. However, still able to explore a good maximum trotter depth to use.
    Using linear regression to analyse the result to get a optimized maximum trotter depth.
    """
    optimizedDepthArray,times = np.load(fileName)[0],np.load(fileName)[1]
    X = times.reshape(-1,1)
    Y = optimizedDepthArray
    slope, _, _, _ = np.linalg.lstsq(X,Y,rcond=None)
    print(f"Optimized Trotter Depth: {slope}")

AdaptingAnalysis(DoDepthOptimize())

#AdaptingAnalysis("Optimized_Depth_mt200_tt1_tr5_X.npy")
#AdaptingAnalysis("Optimized_Depth_mt200_tt1_tr5_Y.npy")
#AdaptingAnalysis("Optimized_Depth_mt200_tt1_tr5_Z.npy")

