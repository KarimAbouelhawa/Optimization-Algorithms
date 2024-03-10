#!/usr/bin/env python
# coding: utf-8



import numpy as np
import random
import math
import sys




def createAreaObstacles (): #create the Air resistance (obstacles) in the area
    array_2d = np.zeros((areaLength, areaWidth), dtype=int)
    n = ((areaLength * areaWidth) // 100) + 2
    for _ in range(n):
        sub_area_rows = random.randint(2, max(int(len(array_2d)/5), 3))
        sub_area_cols = random.randint(2, max(int(len(array_2d[0])/5), 3))

        start_row = random.randint(0, array_2d.shape[0] - sub_area_rows)
        start_col = random.randint(0, array_2d.shape[1] - sub_area_cols)

        array_2d[start_row:start_row + sub_area_rows, start_col:start_col + sub_area_cols] += 1

    return array_2d




numGenerations = 30
colonySize = 10 #num of bees in the colony
initPhermone = 100 #the initial phermone levels per link
minPhermone = 1
p = 0.01 #evaporation
a = 1# power of phermone
b = 0.5 # power of fitness
Q = 10000 #deposition

areaLength = 15 #length of area to be scanned
areaWidth = 15 #width of area to be scanned
areaObst = createAreaObstacles() #the air resistance

w1 = 3 # weight for distance in fitness func
w2 = 5 # weight for delay in fitness func

uavAreaLength = areaLength //2 #length of area to be scanned by one UAV

numOfUavs = 2







def calculateDistance(startPos, endPos): #calculate distance between 2 points
    endWidth = endPos[1]
    endLength = endPos[0]
    startWidth = startPos[1]
    startLength = startPos[0]
    
    diffWidthSq = pow((endWidth - startWidth),2)
    diffLengthSq = pow((endLength - startLength),2)
    
    distance = math.sqrt(diffWidthSq + diffLengthSq)
    
    return distance




def totalDistance (solution):  #calculate total distance for a solution of a UAV
    totalDistance = 0
    delay = 0
    for i in range(len(solution) - 1):
        totalDistance += calculateDistance(solution[i],solution[i+1])
    return totalDistance        
    









def fitnessFunc(solution):
    distance = totalDistance(solution)

    for i in range(len(solution) -1):
        posX, posY = solution[i]
        valueObst = areaObst[posX][posY]
        
        nextX, nextY = solution[i+1]
        nextValueObst = areaObst[nextX][nextY]
        
        diff = valueObst - nextValueObst

        delay = 0
        
        if(valueObst > 0 and nextValueObst > 0):
            delay += 1 + abs(diff)

    fitness = w1*distance + w2*delay
    return round(fitness, 1)




def initStartPos(): # generate random init starting positions
    startPositions = []
    for i in range(colonySize):
        length = random.randint(0, areaLength-1)
        width = random.randint(0, areaWidth-1)
        startPositions.append([length,width])

    return startPositions       




def nextPosFitness(currPos, nextPos): # calculates the fitness for next possible movements
    distance = calculateDistance(currPos, nextPos)

    posX, posY = currPos
    valueObst = areaObst[posX][posY]
    
    nextX, nextY = nextPos
    nextValueObst = areaObst[nextX][nextY]
    
    diff = valueObst - nextValueObst

    delay = 0

    if(valueObst > 0 and nextValueObst > 0):
        delay += 1 + abs(diff)

    fitness = w1*distance + w2*delay
    return fitness
    




def nextPos(currPos, phermones, visited, uavNum): #chooses the next positon the ant will move based on the next pos fitness from current position
    probList = []
    summation = 0
    cumProb = 0
    prob = 0
    r = random.random()
    nextPosition = np.zeros(2)
    if(uavNum == 0): #for uav1

        for i in range(uavAreaLength):
            for j in range(areaWidth):
                if(visited[i][j] != 1):
                    pher = phermones[currPos[0]][currPos[1]][i][j]
                    nextFitness = nextPosFitness(currPos, [i,j])
                    if(nextFitness == 0):
                        nextFitness = 0.1 * w1
                    n = 1/nextFitness
                    multiplication = pow(pher,a) * pow(n, b)
    #                 print("multiplication = ", multiplication)
                    summation += multiplication




        for i in range(uavAreaLength):
            for j in range(areaWidth):
                if(visited[i][j] != 1):
                    pher = phermones[currPos[0]][currPos[1]][i][j]
                    nextFitness = nextPosFitness(currPos, [i,j])
                    if(nextFitness == 0):
                        nextFitness = 0.1 * w1
                    n = 1/nextFitness
                    multiplication = pow(pher,a) * pow(n, b)
                    prob = multiplication / summation
                    probList.append(prob)
                    cumProb += prob
                    if(cumProb >= r):
                        nextPosition = [i,j]
                        return nextPosition
    elif(uavNum == 1): #for uav 2
        
        for i in range(uavAreaLength,areaLength):
            for j in range(areaWidth):
                if(visited[i][j] != 1):
                    pher = phermones[currPos[0]][currPos[1]][i][j]
                    nextFitness = nextPosFitness(currPos, [i,j])
                    if(nextFitness == 0):
                        nextFitness = 0.1 * w1
                    n = 1/nextFitness
                    multiplication = pow(pher,a) * pow(n, b)
    #                 print("multiplication = ", multiplication)
                    summation += multiplication




        for i in range(uavAreaLength,areaLength):
            for j in range(areaWidth):
                if(visited[i][j] != 1):
                    pher = phermones[currPos[0]][currPos[1]][i][j]
                    nextFitness = nextPosFitness(currPos, [i,j])
                    if(nextFitness == 0):
                        nextFitness = 0.1 * w1
                    n = 1/nextFitness
                    multiplication = pow(pher,a) * pow(n, b)
                    prob = multiplication / summation
                    probList.append(prob)
                    cumProb += prob
                    if(cumProb >= r):
                        nextPosition = [i,j]
                        return nextPosition
                    
    print("Error!")     
    print("cumProb = ", cumProb)
    print("r = ", r)
    print("i = ", i)
    print("j = ", j)
    print("currPos[0] = ", currPos[0])
    print("currPos[1] = ", currPos[1])
    print("prob = ", prob)
    print("summation = ", summation)
    
    print(phermones)
    return "ERROR!"




def finalNextPos(currPos, phermones, visited, uavNum): # chooses the next best position but for the final solutiion with the best fitness without any randomness so without exploration
    probList = []
    summation = 0
    prob = 0
    nextPosition = np.zeros(2)
    maxProb = -sys.maxsize + 2
    maxNextPos = []
    
    if(uavNum == 0):
        for i in range(uavAreaLength):
            for j in range(areaWidth):
                if(visited[i][j] != 1):
                    pher = phermones[currPos[0]][currPos[1]][i][j]
                    nextFitness = nextPosFitness(currPos, [i,j])
                    if(nextFitness == 0):
                        nextFitness = 0.1 * w1
                    n = 1/nextFitness
                    multiplication = pow(pher,a) * pow(n, b)
    #                 print("multiplication = ", multiplication)
                    summation += multiplication
                    
        
                    
        for i in range(uavAreaLength):
            for j in range(areaWidth):
                if(visited[i][j] != 1):
                    pher = phermones[currPos[0]][currPos[1]][i][j]
                    nextFitness = nextPosFitness(currPos, [i,j])
                    if(nextFitness == 0):
                        nextFitness = 0.1 * w1
                    n = 1/nextFitness
                    multiplication = pow(pher,a) * pow(n, b)
                    prob = multiplication / summation
                    probList.append(prob)
                    
                    if(prob > maxProb):
                        maxProb = prob
                        maxNextPos = [i,j]

    elif(uavNum == 1):
        for i in range(uavAreaLength,areaLength):
            for j in range(areaWidth):
                if(visited[i][j] != 1):
                    pher = phermones[currPos[0]][currPos[1]][i][j]
                    nextFitness = nextPosFitness(currPos, [i,j])
                    if(nextFitness == 0):
                        nextFitness = 0.1 * w1
                    n = 1/nextFitness
                    multiplication = pow(pher,a) * pow(n, b)
    #                 print("multiplication = ", multiplication)
                    summation += multiplication




        for i in range(uavAreaLength,areaLength):
            for j in range(areaWidth):
                if(visited[i][j] != 1):
                    pher = phermones[currPos[0]][currPos[1]][i][j]
                    nextFitness = nextPosFitness(currPos, [i,j])
                    if(nextFitness == 0):
                        nextFitness = 0.1 * w1
                    n = 1/nextFitness
                    multiplication = pow(pher,a) * pow(n, b)
                    prob = multiplication / summation
                    probList.append(prob)

                    if(prob >= maxProb):
                        maxProb = prob
                        maxNextPos = [i,j]           

    return maxNextPos




def finalSolution(initPos, phermones): #func to generate final solution
    finalAnt = [[],[]]
    finalAnt[0].append(initPos)
    finalAnt[1].append(initPos)
    currPos1 = initPos
    currPos2 = initPos
    visited = np.zeros((areaLength, areaWidth), dtype=int)
    for i in range(uavAreaLength):
        for j in range(areaWidth):
            nextPos = finalNextPos(currPos1, phermones, visited, 0)
            visited[nextPos[0]][nextPos[1]] = 1
            currPos1 = nextPos
            finalAnt[0].append(nextPos)

    for i in range(uavAreaLength, areaLength):
        for j in range(areaWidth):
            nextPos = finalNextPos(currPos2, phermones, visited, 1)
            visited[nextPos[0]][nextPos[1]] = 1
            currPos2 = nextPos
            finalAnt[1].append(nextPos)
            

    fitness1 = fitnessFunc(finalAnt[0])
    fitness2 = fitnessFunc(finalAnt[1])
    fitness = fitness1 + fitness2
    return finalAnt, fitness
    
    




def evaporation(phermones): #calculates the new phermone levels after evaporation
    for i in range(phermones.shape[0]):
        for j in range(phermones.shape[1]):
            for k in range(phermones.shape[2]):
                for l in range(phermones.shape[3]):
                    newPher = phermones[i][j][k][l] * (1-p)
                    phermones[i][j][k][l] = max(minPhermone, newPher)

                    
    return phermones    




def deposition(phermones, allAntsSolution): #calculates new phermone levels after deposition, checks to see where the ants moved and deposits phermones there
    for ant in range(len(allAntsSolution[0])):
        for pos in range(len(allAntsSolution[0][ant]) - 1):
            solDistance1 = totalDistance(allAntsSolution[0][ant])

            deltaPher1 = Q / solDistance1
            
            starti1 = allAntsSolution[0][ant][pos][0]
            startj1 = allAntsSolution[0][ant][pos][1]
            
            endi1 = allAntsSolution[0][ant][pos+1][0]
            endj1 = allAntsSolution[0][ant][pos+1][1]
            
            phermones[starti1][startj1][endi1][endj1] += deltaPher1

            solDistance2 = totalDistance(allAntsSolution[1][ant])

            deltaPher2 = Q / solDistance2

            starti2 = allAntsSolution[1][ant][pos][0]
            startj2 = allAntsSolution[1][ant][pos][1]
            
            endi2 = allAntsSolution[1][ant][pos+1][0]
            endj2 = allAntsSolution[1][ant][pos+1][1]
            
            phermones[starti2][startj2][endi2][endj2] += deltaPher2
            
    return phermones      
    




# import matplotlib.pyplot as plt
# import matplotlib

# #for visualizing 

# matplotlib.use('TkAgg')
# plt.ion()
# def plot_fitness_vs_iteration(data):
#     iterations, fitness_values = zip(*data)
#     plt.plot(iterations, fitness_values, marker='o')
#     plt.xlabel('generation')
#     plt.ylabel('Best Fitness Value')
#     plt.title('Fitness vs. generation')
#     plt.show()




# import matplotlib.pyplot as plt
# import matplotlib



# # matplotlib.use('TkAgg')
# # plt.ion()
def ACO(): #main function
    
    phermones = np.full((areaLength, areaWidth, areaLength, areaWidth), initPhermone) #creates a 4D array to hold the phermone levels between each link
    startPositions = initStartPos() #generates new starting positions
    bestFitnessGeneration = sys.maxsize #the best fitness per generation
    bestFitnessListPerGeneration = []
    visited = np.zeros((areaLength, areaWidth), dtype=int) #checks to see if this position is visited or not
    allAntsSolution = [[],[]] #the solution for all ants
    antSolution = [] #current ant solution
    antSolution2 = []
    #currPos = []
    fitness_vs_iteration = [] #for visualization
    
    for generation in range(numGenerations):
        print("Generation ", generation)
        for ant in range(colonySize):
            print("Ant ", ant)
            antSolution.append(startPositions[ant])
            antSolution2.append(startPositions[ant])
            currPos = startPositions[ant]
            currPos2 = startPositions[ant]
#             visited[startPositions[ant][0]][startPositions[ant][1]] = 1  #Note: you visit the initPos to avoid /0 error
            for i in range(uavAreaLength):
                for j in range(areaWidth):
                    nextPosition = nextPos(currPos, phermones, visited, 0) #chooses the next pos
                    antSolution.append(nextPosition) #adds the chosen nextPos
                    currPos = nextPosition
                    visited[nextPosition[0]][nextPosition[1]] = 1
            
            for i in range(uavAreaLength, areaLength):
                for j in range(areaWidth):
                    nextPosition = nextPos(currPos2, phermones, visited, 1)
                    antSolution2.append(nextPosition)
                    currPos2 = nextPosition
                    visited[nextPosition[0]][nextPosition[1]] = 1
                
            allAntsSolution[0].append(antSolution)
            allAntsSolution[1].append(antSolution2)
            phermones = evaporation(phermones)#does evaporation after each generation
            phermones = deposition(phermones, allAntsSolution) #deposits the phermones
            antFitness1 = fitnessFunc(antSolution)
            antFitness2 = fitnessFunc(antSolution2)
            totalAntFitness = antFitness1 + antFitness2
            if(totalAntFitness < bestFitnessGeneration):
                bestFitnessGeneration = totalAntFitness
                
            antSolution = []
            antSolution2 = []
            visited = np.zeros((areaLength, areaWidth), dtype=int)   
            
        bestFitnessListPerGeneration.append(bestFitnessGeneration)
#         fitness_vs_iteration.append((generation, bestFitnessGeneration))
        allAntsSolution = [[],[]]
        

#     plot_fitness_vs_iteration(fitness_vs_iteration)
#     plt.draw()
#     plt.pause(0.1)  
    bestSolution, bestFitness = finalSolution(startPositions[0], phermones) 
    return bestFitnessListPerGeneration, phermones, bestSolution, bestFitness
            
 









bestFitnessList, phermones, bestSolution, bestFitness = ACO()




print(bestFitnessList)




print(bestFitness)






print(bestSolution[0])




import matplotlib.pyplot as plt

num_iterations = len(bestFitnessList)
iterations = range(1, num_iterations + 1)

plt.figure(figsize=(8, 6))
plt.plot(iterations, bestFitnessList, marker='o', linestyle='-')
plt.title('Fitness Values over Generations')
plt.xlabel('Generation Number')
plt.ylabel('Fitness Value')
plt.grid(True)
plt.show()




###Visualize final path
# get_ipython().run_line_magic('matplotlib', 'notebook')
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # List of positions indicating the path of the vehicle
# path = bestSolution[0]
# fig, ax = plt.subplots()
# line, = ax.plot([], [], marker='o', linestyle='-', color='blue')

# # Set limits and labels
# ax.set_xlim(-1, 7)
# ax.set_ylim(-1, 15)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')

# # Initialize empty arrows
# arrows = [ax.plot([], [], color='red')[0] for _ in range(len(path) - 1)]

# def update(frame):
#     if frame == 0:
#         return line, *arrows

#     x = [point[0] for point in path[:frame]]
#     y = [point[1] for point in path[:frame]]

#     line.set_data(x, y)

#     # Update arrows
#     for i in range(frame - 1):
#         arrows[i].set_data([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]])

#     plt.pause(0.5)  # Pause for 500 milliseconds

#     return line, *arrows

# ani = FuncAnimation(fig, update, frames=len(path) + 1, interval=0, blit=True)
# plt.show()











