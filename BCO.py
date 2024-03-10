import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import math
import copy

noOfGenerations = 10000
colonySize = 50 #num of Bees in a generation 
trailLimit = int(noOfGenerations/10) #number of iterations not changing to discard and replace solution
swapNumber = 30 #number of swaps for solution

onlookerBees = int(colonySize * 0.2) #the number of Onlooker bees

w1 = 3 # weight for distance in fitness func
w2 = 5 # weight for delay in fitness func

areaLength = 15 #length of area to be scanned
areaWidth = 15 #width of area to be scanned

uavAreaLength = areaLength //2 #length of area to be scanned by one UAV

area = np.zeros((areaLength, areaWidth), int)

# -------------------------------------------------------------

def randomSolution (): #Function to generate random solution
    area = np.zeros((areaLength, areaWidth), int)
    list_size = uavAreaLength * areaWidth
    uav1Path = []
    
    for _ in range(list_size):
        random_length = random.randint(0,uavAreaLength-1)
        random_width = random.randint(0,areaWidth-1)
        while area[random_length,random_width] != 0 :
            random_length = random.randint(0,uavAreaLength-1)
            random_width = random.randint(0,areaWidth-1)
        area[random_length,random_width] = 1
        uav1Path.append([random_length,random_width])
    #print(uav1Path)
    #print("---------------------------------------")
    list_size = uavAreaLength * areaWidth
    uav2Path = []

    for _ in range(list_size):
        random_length = random.randint(uavAreaLength, areaLength -1)
        random_width = random.randint(0,areaWidth-1)
        while area[random_length,random_width] != 0 :
            random_length = random.randint(uavAreaLength, areaLength -1)
            random_width = random.randint(0,areaWidth-1)
        area[random_length,random_width] = 1
        uav2Path.append([random_length,random_width])
    #print(uav2Path)
    
    
    return uav1Path,uav2Path

# -------------------------------------------------------------

def calculateDistance(startPos, endPos): #calculate distance between 2 points
    endWidth = endPos[1]
    endLength = endPos[0]
    startWidth = startPos[1]
    startLength = startPos[0]

    diffWidthSq = pow((endWidth - startWidth), 2)
    diffLengthSq = pow((endLength - startLength), 2)

    distance = math.sqrt(diffWidthSq + diffLengthSq)

    return distance

def totalDistance(solution): #calculate total distance for a solution of a UAV
    totalDistance = 0
    for i in range(len(solution) - 1):
        totalDistance += calculateDistance(solution[i], solution[i + 1])
    return totalDistance

# -------------------------------------------------------------

def initStartPos(): # generate random init starting positions
    startPositions = []
    for i in range(colonySize):
        length = random.randint(0, areaLength - 1)
        width = random.randint(0, areaWidth - 1)
        startPositions.append([length, width])

    return startPositions

# -------------------------------------------------------------

def createAreaObstacles(): # create the Air resistance (obstacles) in the area
    array_2d = np.zeros((areaLength, areaWidth), dtype=int)
    n = ((areaLength * areaWidth) // 100) + 2
    for _ in range(n):
        sub_area_rows = random.randint(2, max(int(len(array_2d) / 5), 3))
        sub_area_cols = random.randint(2, max(int(len(array_2d[0]) / 5), 3))

        start_row = random.randint(0, array_2d.shape[0] - sub_area_rows)
        start_col = random.randint(0, array_2d.shape[1] - sub_area_cols)

        array_2d[
            start_row : start_row + sub_area_rows, start_col : start_col + sub_area_cols
        ] += 1

    return array_2d

areaObst = createAreaObstacles()

# -------------------------------------------------------------

def fitnessFunc(solution): #the fitness Func
    distance = totalDistance(solution)

    for i in range(len(solution) - 1):
        posX, posY = solution[i]
        valueObst = areaObst[posX][posY]

        nextX, nextY = solution[i + 1]
        nextValueObst = areaObst[nextX][nextY]

        diff = valueObst - nextValueObst

        delay = 0

        if valueObst > 0 and nextValueObst > 0:
            delay += 1 + abs(diff)

    fitness = w1 * distance + w2 * delay
    return round(fitness, 1)

# ------------------------------------------------------------

def rouletteWheel(currentFitness): # to choose the solution based on cumalitive fitness and random number
    totalFitness = 0
    cumFitness = 0
    randomNum = random.random()
    for i in range(len(currentFitness)):
        totalFitness += currentFitness[i]
    for i in range(len(currentFitness)):
        cumFitness += (currentFitness[i] / totalFitness)
        if cumFitness > randomNum:
            return i
    return (len(currentFitness) - 1)

# -------------------------------------------------------------

def bestFitness(fitnessList): #finds the best fitness in a fitness list
    minFitness = sys.maxsize
    for fitness in fitnessList:
        if fitness < minFitness:
            minFitness = fitness
    return minFitness

# -------------------------------------------------------------

def swapMutation(solution): #Swaps two random positions in the soluton
    
    for _ in range(swapNumber):
        randomPos1 = random.randint(1, len(solution)-1)
        randomPos2 = random.randint(1, len(solution)-1)
        while randomPos1 == randomPos2:
            randomPos2 = random.randint(0, len(solution)-1) 
        
        
        x = solution[randomPos1]
        solution[randomPos1] = solution[randomPos2]
        solution[randomPos2] = x

    return solution

# -------------------------------------------------------------

def scrambleMutation(solution): #scrambles part of the solution
    randomPos1 = random.randint(1, len(solution)-1)
    randomPos2 = random.randint(1, len(solution)-1)
    while randomPos1 == randomPos2:
        randomPos2 = random.randint(0, len(solution)-1) 
    
    if(randomPos2 > randomPos1):
        i2 = randomPos2
        i1 = randomPos1
    else:
        i2 = randomPos1
        i1 = randomPos2
       
    subsetToScramble = solution[i1:i2+1]
    random.shuffle(subsetToScramble)
    
    child = solution[:i1] + subsetToScramble + solution[i2+1:]
        
    
    return child

# -------------------------------------------------------------

def bestFitnessIndex(currentFitness): #returns the index of the best solution based on fitness
    minFitness = sys.maxsize
    index = 0
    for fitness in range(len(currentFitness)):
        if currentFitness[fitness] < minFitness:
            minFitness = currentFitness[fitness]
            index = fitness
    return index
# -------------------------------------------------------------

class Bee: #The Bee class which contains its solutions for the 2 UAVS and their respective fitness and functions needed to update the bee's solution
    def __init__(self, initPos):
        self.solution1 , self.solution2 = randomSolution()

        self.solution1.insert(0,initPos)
        self.solution2.insert(0,initPos)

        self.currentTrail = 0
        self.fitness = fitnessFunc(self.solution1) + fitnessFunc(self.solution2)

    def newRandomSolutions(self, initPos):
        self.solution1 , self.solution2 = randomSolution()

        self.solution1.insert(0,initPos)
        self.solution2.insert(0,initPos)

        self.fitness = fitnessFunc(self.solution1) + fitnessFunc(self.solution2)
        self.currentTrail = 0
    
    def checkSwapSolutions(self, initPos):
        newSolution1 = scrambleMutation(self.solution1)
        newSolution2 = scrambleMutation(self.solution2)

        # newSolution1.insert(0,initPos)
        # newSolution2.insert(0,initPos)

        newFitness = fitnessFunc(newSolution1) + fitnessFunc(newSolution2)
        if newFitness < self.fitness:
            self.solution1 = newSolution1
            self.solution2 = newSolution2
            self.fitness = newFitness
            self.currentTrail = 0
        else:
            self.currentTrail += 1

# -------------------------------------------------------------


def BCO(): #The main function of the BCO that runs everything
    startPositions = initStartPos() #generates random starting positions
    currentFitness = [] #the current fitness values of the generation
    bestGenerationFitness = [] # list of the best fitness in every generation
    hive = [] #list of bees (solutions)
    for i in range(colonySize): #generates bees with random solutions for the colony size
        newBee = Bee(startPositions[i])
        hive.append(newBee)
        currentFitness.append(newBee.fitness)
    
    bestBee = copy.deepcopy(hive[bestFitnessIndex(currentFitness)])
    bestGenerationFitness.append(bestFitness(currentFitness))
    currentFitness = []

    for generation in range(noOfGenerations):
        if generation % 1000 == 0:
            print("Generation ", generation)
        currentFitness = []
        for bee in range(colonySize):
            hive[bee].checkSwapSolutions(startPositions[bee]) #call function in Bee class to swap solution and check if its better, if not better we dont change it
            currentFitness.append(hive[bee].fitness)
        for _ in range(onlookerBees): #the onlooker bees chooses by roulettewheel selection a bee and does swapping again to see if its solution will increase
            index = rouletteWheel(currentFitness)
            hive[index].checkSwapSolutions(startPositions[bee])
            currentFitness[index] = hive[index].fitness
        for bee in range(colonySize): # here any bee that hasnt improved for a number of iterations equal to the trail limit will be dicarded and replaced with new random solution
            if hive[bee].currentTrail > trailLimit:
                newStartPositions = initStartPos()
                hive[bee].newRandomSolutions(newStartPositions[bee])
                currentFitness[bee] = hive[bee].fitness

        currentBestFitness = bestFitness(currentFitness)
        bestGenerationFitness.append(currentBestFitness)
        
        if currentBestFitness < bestBee.fitness: #updates the best performing bee
            # print("best bee before", bestBee.fitness)
            bestBee = copy.deepcopy(hive[bestFitnessIndex(currentFitness)])
            # print("best bee after", bestBee.fitness) 
            # print("current best ", currentBestFitness)

    return bestGenerationFitness, bestBee, currentFitness

bestGenerationFitness, bestBee, currentFitness = BCO()
print("Initial Fitness: ", bestGenerationFitness[0])
print("Best Fitness: ", bestBee.fitness)

num_iterations = len(bestGenerationFitness)
iterations = range(1, num_iterations + 1)

plt.figure(figsize=(8, 6))
plt.plot(iterations, bestGenerationFitness, marker='o', linestyle='-')
plt.title('Fitness Values over Generations')
plt.xlabel('Generation Number')
plt.ylabel('Fitness Value')
plt.grid(True)
plt.show()