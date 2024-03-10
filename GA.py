
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

generationNum = 0

noOfGenerations = 3000 #number of generations
generationSize = 20 #Size of generation

percentElite = 0.2 # 20% for Elite
percentCrossover = 0.6 # 60% for crossover
percentMutation = 0.2 # 20% for mutation

w1 = 3 # weight for distance in fitness func
w2 = 5 # weight for delay in fitness func

crossOver = int(percentCrossover * generationSize) #number of individuals to crossover
elite = int(percentElite * generationSize) #number of individuals than are elite
mutation = int(percentMutation * generationSize) #number of individuals to mutate 



crossoverType = "Davis Order Crossover (OX1)" # we use Davis crossover for crossing over
mutationType = "Scramble Mutation" # we use scramble Mutation to miutate
survivorSelection = "Fitness Based Selection" # we use fitness based selection for the survivors for each generation

areaLength = 15
areaWidth = 15

uavAreaLength = areaLength //2

numOfUavs = 2



def createAreaObstacles (): # Generates obstacles randomly within the area
    array_2d = np.zeros((areaLength, areaWidth), dtype=int)
    n = ((areaLength * areaWidth) // 100) + 2
    for _ in range(n):
        sub_area_rows = random.randint(2, max(int(len(array_2d)/5), 3))
        sub_area_cols = random.randint(2, max(int(len(array_2d[0])/5), 3))

        start_row = random.randint(0, array_2d.shape[0] - sub_area_rows)
        start_col = random.randint(0, array_2d.shape[1] - sub_area_cols)

        array_2d[start_row:start_row + sub_area_rows, start_col:start_col + sub_area_cols] += 1

    return array_2d

areaObst = createAreaObstacles()

def initSolution (): # Randomly initialize UAV paths within the area
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
        uav1Path.append([random_length,random_width]) # Generate initial positions for individuals in the generation
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
        uav2Path.append([random_length,random_width])# Generate initial positions for individuals in the generation
    #print(uav2Path)
    
    
    return [uav1Path,uav2Path]


#[[uav1PathS1,uav2PathS1],.....,[uav1PathSn,uav2PathSn]]

while ((crossOver + elite + mutation) < generationSize):
    #print("h")
    crossOver +=1

initSolutions = []
for i in range(generationSize):
    initSolutions.append(initSolution())
initPosition=[]
for _ in range(generationSize):
    randomX = random.randint(0, areaWidth-1)
    randomY = random.randint(0, areaLength-1)
    initPosition.append([randomX,randomY])   
# print(len(initSolutions))  
# print(initPosition)


# Calculate the distance between two points
def calculateDistance(startPos, endPos):
    endWidth = endPos[1]
    endLength = endPos[0]
    startWidth = startPos[1]
    startLength = startPos[0]
    
    diffWidthSq = pow((endWidth - startWidth),2)
    diffLengthSq = pow((endLength - startLength),2)
    
    distance = math.sqrt(diffWidthSq + diffLengthSq)
    
    return distance
    
# Calculate total distance in a solution path
def totalDistance (solution):
    totalDistance = 0
    for i in range(len(solution) - 1):
        totalDistance += calculateDistance(solution[i],solution[i+1])
    return totalDistance        
    
# Davis Order Crossover function for genetic operations
def davisCrossOver(solution1 , solution2):
    child1 = [0 for _ in range(len(solution1))]
    child2 = [0 for _ in range(len(solution1))]
    randomPos1 = random.randint(0, len(solution1)-1)
    randomPos2 = random.randint(0, len(solution1)-1)
    while randomPos1 == randomPos2:
        randomPos2 = random.randint(0, len(solution1)-1) 
    
    if(randomPos2 > randomPos1):
        i2 = randomPos2
        i1 = randomPos1
    else:
        i2 = randomPos1
        i1 = randomPos2
               

    
    #print("i1 is " , i1)
    #print("i2 is " , i2)
    for i in range (i2 - i1 + 1):
        currI = i1 + i
        child1[currI] = solution1[currI]
        child2[currI] = solution2[currI]
        
    indexSol1 = i2 +1
    indexChild1 = i2 +1
    
    while(0 in child1):
        if(indexSol1 == len(solution1)):
                indexSol1 = 0  
        if(indexChild1 == len(solution1)):
                indexChild1 = 0  
        if (solution2[indexSol1] not in child1):
            child1[indexChild1] = solution2[indexSol1]
            indexSol1 +=1
            indexChild1 +=1    
            
        else:
            indexSol1 +=1
            
            
            
            
    indexSol2 = i2 +1
    indexChild2 = i2 +1
    
    while(0 in child2):
        if(indexSol2 == len(solution1)):
                indexSol2 = 0  
        if(indexChild2 == len(solution1)):
                indexChild2 = 0  
        if (solution1[indexSol2] not in child2):
            child2[indexChild2] = solution1[indexSol2]
            indexSol2 +=1
            indexChild2 +=1    
            
        else:
            indexSol2 +=1
                     
  
    
    return child1,child2
    

# child1,child2,i1,i2 = davisCrossOver ([[0,0],[0,4],[0,2],[0,3],[0,1],[0,6],[0,5],[0,7],[0,9],[0,8]]
#                                       ,[[0,1],[0,0],[0,2],[0,3],[0,4],[0,8],[0,9],[0,6],[0,7],[0,5]])
# print(i1)
# print(i2)
# print (child1)
# print (child2)


# swap Mutation function for genetic operations
def swapMutation(solution):
    
    
    randomPos1 = random.randint(0, len(solution)-1)
    randomPos2 = random.randint(0, len(solution)-1)
    while randomPos1 == randomPos2:
        randomPos2 = random.randint(0, len(solution)-1) 
    
    
    x = solution[randomPos1]
    solution[randomPos1] = solution[randomPos2]
    solution[randomPos2] = x

    return solution
    
# Scramble Mutation function for genetic operations
def scrambleMutation(solution):
    randomPos1 = random.randint(0, len(solution)-1)
    randomPos2 = random.randint(0, len(solution)-1)
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
    

#def fitnessFunc(distance, speed):
#    fitness = w1 * (distance/speed) + w2 * cost * speed 
#    return round(fitness, 1)


# Fitness function calculation considering distance and delay due to air resistance
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



def GA (initSolution,crossOver,elite,mutation,initPositions):
    fitnessLevels = [] #fitness level list per generation
    distance = 0
    newSolution = []
    crossOverChildren =[] #List for the children of crossover from the population
    newInitPositions = []
    for i in range (len(initSolution)):
        fitnessLevels.append(fitnessFunc(initSolution[i][0]) + fitnessFunc(initSolution[i][1])) #add initial fitness levels
                                                                        
    sortedIndices = sorted(range(len(fitnessLevels)), key=lambda i: fitnessLevels[i]) #sort to find the elite in population
    
    mutationIndices = sorted(range(len(fitnessLevels)), key=lambda i: fitnessLevels[i])[-mutation:]  #choose the worst performing solutions
    
    for i in range(elite): #add the elite solutions to the next generation
        newSolution.append(initSolution[sortedIndices[i]])
        newInitPositions.append(initPositions[sortedIndices[i]])
        
        
    sortedX = sorted(zip(fitnessLevels, initSolution))
    _, sortedSolutions = zip(*sortedX)    
    
    
    parentsCrossed = []
    numChildren = 0
    while(numChildren < crossOver): #loop to get crossover children
        
        parent1Index = random.randint(0, len(initSolution)-1 - mutation) #choose parents
        parent2Index = random.randint(0, len(initSolution)-1 - mutation) #choose parents
        parents = [parent1Index,parent2Index] # parents chosen
        while(parent1Index == parent2Index or parents in parentsCrossed): #making sure they were not crossed before
            parent1Index = random.randint(0, len(initSolution)-1)
            parent2Index = random.randint(0, len(initSolution)-1)
            parents = [parent1Index,parent2Index]
        parentsCrossed.append([parent1Index,parent2Index])
        parentsCrossed.append([parent2Index,parent1Index])
        
        parent1 = sortedSolutions[parent1Index]
        parent2 = sortedSolutions[parent2Index]
        
        child1Uav1, child2Uav1 = davisCrossOver(parent1[0],parent2[0]) #cross using davis cross over for uav1
        child1Uav2, child2Uav2 = davisCrossOver(parent1[1],parent2[1]) #cross using davis cross over for uav2
        crossOverChildren.append([child1Uav1,child1Uav2])
        crossOverChildren.append([child2Uav1,child2Uav2])
        

        #crossOver the initPositions to get children init positions using arithmatic crossingover


        parent1InitPos = initPositions[parent1Index]
        parent2InitPos = initPositions[parent2Index]
        
        a=0.7
        
        child1X = a * parent1InitPos[0] + (1-a)*parent2InitPos[0]
        child1y = a * parent1InitPos[1] + (1-a)*parent2InitPos[1]
        
        child2X = a * parent2InitPos[0] + (1-a)*parent1InitPos[0]
        child2y = a * parent2InitPos[1] + (1-a)*parent1InitPos[1]
        
        child1Pos = [int(child1X),int(child1y)]
        child2Pos = [int(child2X),int(child2y)]
        
        newInitPositions.append(child1Pos)
        newInitPositions.append(child2Pos)
        

       
#         averageX = int((parent1InitPos[0] + parent2InitPos[0])/2)
#         averageY = int((parent1InitPos[1] + parent2InitPos[1])/2)
#         newInitPositions.append([averageX,parent2InitPos[1]])
#         newInitPositions.append([parent2InitPos[0],averageY])

#         newInitPositions.append([parent1InitPos[0],parent2InitPos[1]])
#         newInitPositions.append([parent2InitPos[0],parent1InitPos[1]])
        numChildren +=2
        
        
    for i in range(crossOver): #add the children to next generation
        newSolution.append(crossOverChildren[i])
    
    for i in range (mutation): #loop to get the mutated population in next generation
        mutationUav1 = scrambleMutation(initSolution[mutationIndices[i]][0])
        mutationUav2 = scrambleMutation(initSolution[mutationIndices[i]][1])
        newSolution.append([mutationUav1,mutationUav2])
        randomX = random.randint(0, areaWidth-1)
        randomY = random.randint(0, areaLength-1)
        newInitPositions.append([randomX,randomY])
        
                                                                        

    return newSolution,newInitPositions                
    

# newSolution = GA(initSolutions,speedList,crossOver,elite,mutation)

# len(newSolution)

def bestFitness(newSolution):
    fitnessLevels = []
    for i in range (len(newSolution)):
        fitnessLevels.append(fitnessFunc(newSolution[i][0]) + fitnessFunc(newSolution[i][1]))
    eliteIndices = sorted(range(len(fitnessLevels)), key=lambda i: fitnessLevels[i])[:elite]
    print("Best fitness Func = ",fitnessLevels[eliteIndices[0]])



def printFitnessFunctions(newSolution):
    fitnessLevel = []
    for i in range (len(newSolution)):
        fitnessLevel.append(fitnessFunc(newSolution[i][0]) + fitnessFunc(newSolution[i][1]))
        
    for i in range(len(newSolution)):
        print("fitness Func os sol ",i," = ",fitnessLevel[i] )
        



matplotlib.use('TkAgg')
plt.ion()

# Initialize empty lists to store the history
generation_history = []
fitness_history = []

def visualizeEliteFitness(newSolution, newInitPositions, generationNum):
    fitness =fitnessFunc(newSolution[0][0]) + fitnessFunc(newSolution[0][1])

    # Append the current generation number and fitness value to the history
    generation_history.append(generationNum)
    fitness_history.append(fitness)

    plt.clf()  # Clear the previous plot
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Value vs. Generation Number')

    # Plot the entire history of points
    plt.plot(generation_history, fitness_history, marker='o', linestyle='-', label='Elite Fitness')

    plt.legend()  # Show legend if you added labels
    plt.grid(True)  # Show grid for better readability
    plt.pause(0.5)

    print("Current elite fitness value =", fitness)
    plt.show()  # Display the figure





newSolution = initSolutions[:]
newInitPositions = initPosition
generationNum = 0
for _ in range (noOfGenerations):
    if(generationNum % 1000 == 0):
        print("!!!!!!Generation Num = ",generationNum)
    
    generationNum +=1
    newSolution,newInitPositions = GA(newSolution,crossOver,elite,mutation,newInitPositions)
    #bestFitness(newSolution)
    
    if(generationNum % 100 == 0 ):
        visualizeEliteFitness(newSolution,newInitPositions,generationNum)
    
  





fitnessLevels = []
for i in range (len(newSolution)):
        fitnessLevels.append(fitnessFunc(newSolution[i][0]) + fitnessFunc(newSolution[i][1]))
eliteIndices = sorted(range(len(fitnessLevels)), key=lambda i: fitnessLevels[i])[:elite]
# print(newSolution[eliteIndices[0]][0])
print(newInitPositions[eliteIndices[0]])
print("final fitness Func = ",fitnessLevels[eliteIndices[0]])

###Visualize final path

# List of positions indicating the path of the vehicle
path = newSolution[eliteIndices[0]][0]
fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o', linestyle='-', color='blue')

# Set limits and labels
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 15)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Initialize empty arrows
arrows = [ax.plot([], [], color='red')[0] for _ in range(len(path) - 1)]

def update(frame):
    if frame == 0:
        return line, *arrows

    x = [point[0] for point in path[:frame]]
    y = [point[1] for point in path[:frame]]

    line.set_data(x, y)

    # Update arrows
    for i in range(frame - 1):
        arrows[i].set_data([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]])

    # plt.pause(0.5)  # Pause for 500 milliseconds

    return line, *arrows

ani = FuncAnimation(fig, update, frames=len(path) + 1, interval=0, blit=True)
plt.show()




# ####Visualize initial path

# # %matplotlib notebook
# # import matplotlib.pyplot as plt
# # from matplotlib.animation import FuncAnimation

# # List of positions indicating the path of the vehicle
# path = initSolutions[0][0]
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




fitnessLevels = []
for i in range (len(initSolutions)):
        fitnessLevels.append(fitnessFunc(initSolutions[i][0]) + fitnessFunc(initSolutions[i][1]))
eliteIndices = sorted(range(len(fitnessLevels)), key=lambda i: fitnessLevels[i])[:elite]
# print(initSolutions[eliteIndices[0]][0])
print("initial fitness Func = ",fitnessLevels[eliteIndices[0]])    
#print("fitness Func = ",fitnessLevels)




