#!/usr/bin/env python
# coding: utf-8



import numpy as np
import random
import math
import sys



iTemp = 500 #initial temp   
fTemp = 0.01 #final temp
cTemp = iTemp #current temp
iMax = 200
b=2.5
bVariable = ((fTemp - iTemp)/iMax) * -1

#b=2.5

#b = ((fTemp - iTemp)/iMax) * -1 #i will minus it to the Temp in main func

numIter = 1
#maxNumUavs = 5

w1 = 3 # weight for distance in fitness func
w2 = 5 # weight for delay in fitness func


# Define area dimensions and create obstacles in the area
areaLength = 15 #length of area to be scanned
areaWidth = 15 #width of area to be scanned

uavAreaLength = areaLength //2  #length of area to be scanned by one UAV

area = np.zeros((areaLength, areaWidth), int)

swapNumber = 10






print(b)



# create the Air resistance (obstacles) in the area
def createAreaObstacles():
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

#Function to generate random solution
def randomSolution ():
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

#calculate distance between 2 points
def calculateDistance(startPos, endPos):
    endWidth = endPos[1]
    endLength = endPos[0]
    startWidth = startPos[1]
    startLength = startPos[0]

    diffWidthSq = pow((endWidth - startWidth), 2)
    diffLengthSq = pow((endLength - startLength), 2)

    distance = math.sqrt(diffWidthSq + diffLengthSq)

    return distance

# generate initail random starting positions
def initStartPos():
    length = random.randint(0, areaLength - 1)
    width = random.randint(0, areaWidth - 1)
    startPosition = [length, width]

    return startPosition

 #calculate total distance for a solution of a UAV
def totalDistance(solution):
    totalDistance = 0
    for i in range(len(solution) - 1):
        totalDistance += calculateDistance(solution[i], solution[i + 1])
    return totalDistance


def fitnessFunc(solution):#the fitness Func
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


def bestFitness(fitnessList):
    minFitness = sys.maxsize
    for fitness in fitnessList:
        if fitness < minFitness:
            minFitness = fitness
    return minFitness

#Swaps between two random positions
def swap(solution):
    
    for _ in range(swapNumber):
        randomPos1 = random.randint(1, len(solution)-1)
        randomPos2 = random.randint(1, len(solution)-1)
        while randomPos1 == randomPos2:
            randomPos2 = random.randint(0, len(solution)-1) 
        
        
        x = solution[randomPos1]
        solution[randomPos1] = solution[randomPos2]
        solution[randomPos2] = x

    return solution

#Compare the probability and random number to see if the prob is higher it will take the worst solution(exploration if temp is relatively high)
def probCompare(bestFitness, currentFitness, cTemp):
    deltaF= currentFitness - bestFitness
    prob = math.exp(-deltaF/cTemp)
    randomNum =  random.random()
    return prob > randomNum


def SA():
    global cTemp
    uav1Path,uav2Path = randomSolution () #generates random solutions
    startPos = initStartPos() #generate initPositions
    uav1Path.insert(0, startPos)
    uav2Path.insert(0, startPos)
    bestFitness = fitnessFunc(uav1Path) + fitnessFunc(uav2Path) #fitness function for inital solution
    fitnessList = [bestFitness]
    
    i = 1
    while(cTemp > fTemp): #while current temperature is higher than final temperature
        newUav1 = swap(uav1Path) #do swapping of the solution of uav1
        newUav2 = swap(uav2Path) #do swapping of the solution of uav2
        
        newFitness = fitnessFunc(newUav1) + fitnessFunc(newUav2) #get new fitness function
        if((newFitness < bestFitness) or (probCompare(bestFitness,newFitness,cTemp))): #check whether to take this new solution if it either better or the probability is higher than the random number
            #replace the new path
            uav1Path = newUav1 
            uav2Path = newUav2
            bestFitness = newFitness
            fitnessList.append(bestFitness) #add the fitness to the list
        else:
            fitnessList.append(bestFitness)
            
        cTemp = iTemp - (b * i) #reduce the current temperature LINEARLY using b coeffiecient
        i = i + 1
    cTemp = iTemp
    return uav1Path, uav2Path, bestFitness, fitnessList


uav1Path, uav2Path, bestFitness, fitnessList = SA()



print(uav1Path[0])
print(fitnessList)




print(fitnessList[0])




print(fitnessList[-1])




import matplotlib.pyplot as plt

num_iterations = len(fitnessList)
iterations = range(1, num_iterations + 1)

plt.figure(figsize=(8, 6))
plt.plot(iterations, fitnessList, marker='o', linestyle='-')
plt.title('Fitness Values over Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Fitness Value')
plt.grid(True)
plt.show()
