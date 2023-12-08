import numpy as np
from tkinter import *
import random
import math
import cProfile
import RaindropNeuralNetwork as RNN
from RaindropCanvas import *
import copy
import multiprocessing as mp

# This variable determines how fast the goalpost moves
global goalPostChange
goalPostChange = 2

# Starting direction of goal (True = right, False = left)
global goingRight
goingRight = True

# Starting point for the goal
global goal_x,goal_x2
goal_x,goal_x2 = 250,350

# Keeps track of all the scores of the neural networks
scoreList = []

# This gets the average tick amount to fall to the bottom
globalTickNum = 0
tickNumCount = 0

def getGoalNum():
    if goingRight: return 1 
    else: return 0
    
def getTickNum(past_goal_x, current_goal_x, past_direction, current_direction):
    if (past_direction == 1):
        if (current_direction == 1):
            # Gets the number of ticks based on the given info
            return (current_goal_x - past_goal_x)/goalPostChange
        else:
            # This is the max value that the goal_x input can be
            maxX = (600 - (goal_x2-goal_x)/2)
            
            # This adjustes for if the goal switched direction
            adjusted_goal_x = maxX + abs(maxX - current_goal_x)
        
            # Gets the number of ticks based on the given info
            return (adjusted_goal_x - past_goal_x)/goalPostChange
    else:
        if (current_direction == 0):
            return (past_goal_x - current_goal_x)/goalPostChange
        else:
            # This is the max value that the goal_x input can be
            minX = (goal_x2-goal_x)/2
        
            # This formula gets the future position of the goal if it is moving left
            return ((past_goal_x + current_goal_x) - minX*2)/goalPostChange
    
def futureGoalNum(current_goal_x, direction, ticknum):
    # The change in x using how much x changes each tick * the amount of ticks
    x_change = goalPostChange*ticknum    

    if (direction == 1):
        # This is the max value that the goal_x input can be
        maxX = (600 - (goal_x2-goal_x)/2)
        
        # This formula checks to see if the goal changed directions
        if (maxX - (current_goal_x + x_change)) < 0:
            return 0
        else: return 1
    else:
        # This is the max value that the goal_x input can be
        minX = (goal_x2-goal_x)/2
        
        # This formula checks to see if the goal changed directions
        if ((current_goal_x - minX) - x_change) < 0:
            return 1
        else: return 0

# TEST THIS FUNCTION AND FIX
def futureGoal(current_goal_x, direction, ticknum):
    # The change in x using how much x changes each tick * the amount of ticks
    x_change = goalPostChange*ticknum    

    if (direction == 1):
        # This is the max value that the goal_x input can be
        maxX = (600 - (goal_x2-goal_x)/2)
        
        # This formula gets the future position of the goal if it is moving right
        return maxX - abs(maxX - (current_goal_x + x_change))
    else:
        # This is the max value that the goal_x input can be
        minX = (goal_x2-goal_x)/2
        
        # This formula gets the future position of the goal if it is moving left
        return abs((current_goal_x - minX) - x_change) + minX
    
def getGoalValue(goalNum1,goalNum2,goal_x_start,goal_x_current):
    if (goalNum1 == goalNum2):
        return goalNum1
    elif (goalNum1 == 1):
        # This is the max value that the goal_x input can be
        maxX = (600 - (goal_x2-goal_x)/2)
        
        # These are the distances from the border
        oldDistanceX = maxX - goal_x_start
        newDistanceX = maxX - goal_x_current
        
        # Gets our denominator
        biggerValue = max(oldDistanceX,newDistanceX)
        
        # This is the final percent value (range -.5 to .5 so adding .5 moves it to 0 to 1)
        return (float(oldDistanceX - newDistanceX)/float(biggerValue))/2.0 + .5 
    else:
        # This is the max value that the goal_x input can be
        minX = (goal_x2-goal_x)/2
        
        # These are the distances from the border
        oldDistanceX = goal_x_start - minX
        newDistanceX = goal_x_current - minX
        
        # Gets our denominator
        biggerValue = max(oldDistanceX,newDistanceX)
        
        # This is the final percent value (range -.5 to .5 so adding .5 moves it to 0 to 1)
        return (float(newDistanceX - oldDistanceX)/float(biggerValue))/2.0 + .5 


def update(drops, score, neural_network, usableInput=None, usableReturn=False):
    # Update the goal
    global vectorRects
    global goal_x,goal_x2
    global globalTickNum, tickNumCount
    goal_x,goal_x2 = moveGoal(goal_x,goal_x2)
    
    # Gets right or left and converts it
    goalNum = getGoalNum()

    # Gets x_coor from network
    if neural_network != None: 
        currentGoal = (goal_x+goal_x2)/2
        
        # Gets the goal Num
        newGoalNum = getGoalValue(goalNum,futureGoalNum(currentGoal, goalNum, int(globalTickNum)),currentGoal,futureGoal(currentGoal, goalNum, int(globalTickNum)))
        NN_x_coor = neural_network.predict(([(goal_x+goal_x2)/1200,newGoalNum]))[0][0]*600
    #print(f"Coor: {NN_x_coor}")

    # This adds more drops
    if (VISUALIZE):
        drop,x_coor = addRandomDrop(False, VISUALIZE, cvs) # xcoor is the x coordinate at which the drop spawns
        #drops = np.append(drops,[[0.0,0.0,x_coor,0.0,drop]], axis=0)
        if neural_network != None:
            drops = np.append(drops,[[0.0,0.0,NN_x_coor,0.0,(goal_x+goal_x2),NN_x_coor,goalNum,drop]], axis=0) # Current X Velocity, Current Y Velocity, Current X Position, Current Y Position, goal_x at drop time (Used for neural network), original drop x (Used for neural network),goalNum, Drop Object
        else:
            drops = np.append(drops,[[0.0,0.0,x_coor,0.0,(goal_x+goal_x2),x_coor,goalNum,drop]], axis=0) # Current X Velocity, Current Y Velocity, Current X Position, Current Y Position, goal_x at drop time (Used for neural network), original drop x, goalNum, Drop Object
        
        # This moves the goalpost on the screen
        global goal
        cvs.moveto(goal, goal_x, 600)
    else:
        if neural_network != None:
            drops = np.append(drops,[[0.0,0.0,NN_x_coor,0.0,(goal_x+goal_x2),NN_x_coor,goalNum]], axis=0) # Current X Velocity, Current Y Velocity, Current X Position, Current Y Position, goal_x at drop time (Used for neural network), original drop x (Used for neural network), goalNum, Drop Object
        else:
            # Get the x coor if we are not doing visualize
            x_coor = random.randint(0,596)
            drops = np.append(drops,[[0.0,0.0,x_coor,0.0,(goal_x+goal_x2),x_coor,goalNum]], axis=0) # Current X Velocity, Current Y Velocity, Current X Position, Current Y Position, goal_x at drop time (Used for neural network), original drop x, goalNum, Drop Object

    # This changes the canvas
    limit = len(drops)
    dropNum = 0
    while dropNum < limit:
        if ((not -.1 < drops[dropNum,3] < 600) or (not -.1 < drops[dropNum,2] < 600)):
            # Iterates Score
            if (goal_x < drops[dropNum,2] < goal_x2) and (drops[dropNum,3] > 600):
                # The score gets lower the farther from the center of the goal
                score += 1 - (abs((goal_x + goal_x2)/2 - drops[dropNum,2])/(goal_x2 - goal_x))
                #print(f"Score: {score} \nScore Remover: {(abs((goal_x + goal_x2)/2 - drops[dropNum,2])/(goal_x2 - goal_x))}")
                diff = goal_x2-goal_x
                if usableReturn and (goal_x+diff/3 < drops[dropNum,2] < goal_x2+diff/3): 
                    # If the goal changed directions, calculate a new value rather than 0 or 1
                    goalScoreNum = getGoalValue(drops[dropNum, 6],goalNum,drops[dropNum, 4]/2,(goal_x+goal_x2)/2)
                    if (goalScoreNum != 0 and goalScoreNum != 1):
                        globalTickNum = ((globalTickNum*tickNumCount) + getTickNum(drops[dropNum, 4]/2, (goal_x+goal_x2)/2, drops[dropNum, 6], goalNum))/(tickNumCount+1)
                        tickNumCount += 1
                    usableInput = np.append(usableInput,[[drops[dropNum, 4]/1200], [goalScoreNum], [drops[dropNum, 5]/600]], axis=1) # /1200 is to get it between 1 & 2, this is goal_x + goal_x2

            
            if (VISUALIZE):
                # Deletes drop from canvas
                cvs.delete(int(drops[dropNum, 7]))
            
            # Deletes drop from array
            drops = np.delete(drops, dropNum, 0)
            
            # Shows that there is one less drop to calculate
            limit -= 1
        else:
            currentRect = vectorRects[math.floor((drops[dropNum,2]/(600/size))),math.floor((drops[dropNum,3]/(600/size)))]
            drops[dropNum, 1] += .2 - currentRect[1] # Positive is down in tkinter so to simulate pyhsics we subtract instead of add
            drops[dropNum, 0] += .0 + currentRect[0]
                
            # This ticks the program without ticking tkinter
            drops = compTick(drops, dropNum)
            
            if (VISUALIZE):
                # Deletes drop from canvas
                cvs.tick(drops[dropNum])
            dropNum += 1
    if usableReturn: return drops,usableInput
    return drops,score

def moveGoal(current_x1, current_x2):
    global goingRight
    # Changes direction if the post gets too close to the edges
    if current_x2 >= 600:
        goingRight = False
    elif current_x1 <= 0:
        goingRight = True
    
    # This maintains current direction
    if goingRight:
        return current_x1+goalPostChange, current_x2+goalPostChange
    else:
        return current_x1-goalPostChange, current_x2-goalPostChange
    

def compTick(drops, dropNum):
    drops[dropNum, 3] += drops[dropNum, 1]
    drops[dropNum, 2] += drops[dropNum, 0]
    
    return drops

# This section makes sure that the value of the horizontal does not exceed .25 and the value of the vertical does not exceed .5
def doesNotExceedBounds(xcomp, ycomp):
    if (xcomp > .5/xNumOfDivs): # Horizontal Vector Component (0)
        xcomp = .5/xNumOfDivs
    elif (xcomp < -.5/xNumOfDivs): # Horizontal Vector Component (0)
        xcomp = -.5/xNumOfDivs
    if (ycomp > .5/yNumOfDivs): # Horizontal Vector Component (1)
        ycomp = .5/yNumOfDivs
    elif (ycomp < -.5/yNumOfDivs): # Horizontal Vector Component (1)
        ycomp = -.5/yNumOfDivs
    return [xcomp, ycomp]
    

# This function inherits vectors from other active squares
def findVectorRect(vectorRects, xDirInt, yDirInt, xNumOfDivs, yNumOfDivs, baseVector):

    base = 2.4
    global size

    rand = random.random()

    if (xDirInt == 0 and yDirInt == 0):                    
        [xDirInt, yDirInt] = baseVector
    elif (yDirInt == 0):
        # This is the everage of the boxes to the left
        horizontalBase = (vectorRects[xDirInt-1, yDirInt, 0]*base + vectorRects[xDirInt-1, yDirInt+1, 0])/(2+(base-1))
        # This is the average of the bozes above
        verticalBase = vectorRects[xDirInt-1, yDirInt+size-1, 1]
        
        [xDirInt, yDirInt] = [horizontalBase + (rand-.5)/(xNumOfDivs*minCumulitiveFactor), verticalBase + (rand-0.5)/(yNumOfDivs*minCumulitiveFactor)] #Gives random vector components to each square 
    elif (xDirInt == 0):
        # This is the average of the bozes above
        verticalBase = vectorRects[xDirInt, yDirInt-1, 1]        

        [xDirInt, yDirInt] = [baseVector[0] + (rand-.5)/(xNumOfDivs*minCumulitiveFactor), verticalBase + (rand-0.5)/(yNumOfDivs*minCumulitiveFactor)] #Gives random vector components to each square
    else:
        # This is the everage of the boxes to the left
        if (yDirInt != size-1):
            horizontalBase = (vectorRects[xDirInt-1, yDirInt-1, 0] + vectorRects[xDirInt-1, yDirInt, 0]*base + vectorRects[xDirInt-1, yDirInt+1, 0])/(3+(base-1))
        else:
            horizontalBase = (vectorRects[xDirInt-1, yDirInt-1, 0] + vectorRects[xDirInt-1, yDirInt, 0]*base)/(2+(base-1))
        # This is the average of the bozes above
        verticalBase = (vectorRects[xDirInt-1, yDirInt-1, 1] + vectorRects[xDirInt, yDirInt-1, 1]*base)/(2+(base-1))
        
        [xDirInt, yDirInt] = [horizontalBase + (rand-.5)/(xNumOfDivs*minCumulitiveFactor), verticalBase + (rand-0.5)/(yNumOfDivs*minCumulitiveFactor)] #Gives random vector components to each square

    return [xDirInt, yDirInt]

# This function initializes the vector squares
def getVectorRects():
    # Starting Square Div
    baseVector = [(random.random()-.5)/xNumOfDivs, (random.random()-.5)/yNumOfDivs]

    newVectorRects = np.zeros((size,size,2), np.float32)

    for xDirInt in range(len(newVectorRects)):
        for yDirInt in range(len(newVectorRects[xDirInt])):   
            # This gets the inheritence from the other squares
            newVectorRects[xDirInt, yDirInt] = findVectorRect(newVectorRects, xDirInt, yDirInt, xNumOfDivs, yNumOfDivs, baseVector)
            newVectorRects[xDirInt, yDirInt] = doesNotExceedBounds(newVectorRects[xDirInt, yDirInt, 0], newVectorRects[xDirInt, yDirInt, 1])
            
    return newVectorRects
    
# User Commands for the Visual
def addDropBlur(e):
    global repeat
    global drops
    drop,x_coor = addRandomDrop(cvs, True)
    drops = np.append(drops, [[0.0,0.0,x_coor,0.0,(goal_x+goal_x2),x_coor,drop]], axis=0)
    repeat = root.after(2, addDropBlur, e)
    
def userScene(cvs):
    root.bind('<Motion>',callback)
    root.bind('<ButtonRelease-1>', stopRepeatDrop)
    root.bind("<Button-1>", addDropBlur)

def stopRepeatDrop(e):
    global repeat
    root.after_cancel(repeat)
    repeat = None

def changeToUser(canvas):
    global generatedDrops
    generatedDrops = 0
    respawnVectorRects()
    userScene(canvas)

def changeToGenerated(canvas):
    global generatedDrops
    generatedDrops = 1
    respawnVectorRects()
    generatedScene(canvas)
    
def respawnVectorRects():
    global vectorRectsObjects
    global cvs
    global vectorRects
    global drops
    
    for objectAndText in vectorRectsObjects:
        cvs.delete(objectAndText[0])
        cvs.delete(objectAndText[1])
        
    vectorRects = getVectorRects()
        
    # Remove all drops
    counter = len(drops)
    while 0 != counter:
        cvs.delete(int(drops[counter-1, 7]))
        np.delete(drops, counter-1, axis=0)
        counter -= 1
        drops = np.array([[0.0, 0.0, 0.0, 0.0, (goal_x+goal_x2), 0.0, cvs.create_oval(0, 0, 2, 2, fill="#190482")], [0.0, 0.0, 50.0, 0.0, (goal_x+goal_x2), 50.0, cvs.create_oval(50, 50, 52, 52, fill="#190482")]], np.float32)
    vectorRectsObjects = getVectorRectVisuals(cvs, vectorRects)
    
def initializeNeuralNetwork(alpha=0.5):
    return RNN.RaindropNueralNetwork([2,3,3,1], alpha=alpha)
  
def main(vectorRectsInput, visualize, simple_network=None, usableInputReturn=False, numOfTicks=800): 
    global vectorRects
    vectorRects = vectorRectsInput
    global drops
    score = 0
    
    # Starting direction of goal (True = right, False = left)
    global goingRight
    goingRight = True

    # Starting point for the goal
    global goal_x,goal_x2
    goal_x,goal_x2 = 250,350
    
    # Changes goalNum
    if goingRight: goalNum = 1 
    else: goalNum = 0
    
    # This sets up the usable input array if we want one
    if usableInputReturn: usableInput = np.empty((3,1), np.float16)
    
    # If this variable is yes, the visualization will pop up, otherwise no.
    global VISUALIZE
    VISUALIZE = visualize
    
    if(VISUALIZE):
    
        # Tkinter stuff
        global vectorRectsObjects
        global cvs
        global root
        cvs,vectorRectsObjects,root = InitializeCanvas(vectorRects) # This returns the canvas and the visual representation of the vector rectangles
        
        userModeButton = Button(root, text="UserMode", command=functools.partial(changeToUser, canvas=cvs))
        userModeButton.place(x=610, y=5)
    
        generatedModeButton = Button(root, text="GeneratedMode", command=functools.partial(changeToGenerated, canvas=cvs))
        generatedModeButton.place(x=610, y=40)
    
        # Goal
        global goal
        goal = cvs.create_rectangle(goal_x,600,goal_x2,620, fill='#5593e8', outline='#5593e8')
    
        # Root binds
        root.bind("<e>", lambda _: respawnVectorRects())
    
        # Initial drops configuration
        drops = np.array([[0.0, 0.0, 0.0, 0.0, goal_x, 0.0, goalNum, cvs.create_oval(0, 0, 2, 2, fill="#190482")], [0.0, 0.0, 50.0, 0.0, goal_x, 50.0, goalNum, cvs.create_oval(50, 50, 52, 52, fill="#190482")]], np.float32)

    else:
        drops = np.array([[0.0, 0.0, 0.0, 0.0, goal_x, 0.0, goalNum], [0.0, 0.0, 50.0, 0.0, goal_x, 50.0, goalNum]], np.float32)


    # Update loop
    i = 0
    while i<numOfTicks:
        # Updates the function
        if usableInputReturn:
            drops,usableInput = update(drops, score, simple_network, usableInput=usableInput, usableReturn=usableInputReturn)
        else:
            drops,score = update(drops, score, simple_network)
        if(not VISUALIZE):
            i += 1
            continue
        
        # Does the required visual updates
        root.update_idletasks()
        if i%4 == 0:
            root.update()
        i += 1
    if VISUALIZE: root.destroy()
    # This is for the section of the neural network in which we gather data
    if usableInputReturn: return usableInput
    return score

def runRandom(simple_network, usableInput, usableOutput, vectorRects):
    # Gets Scorelist
    global scoreList
    
    new_network = copy.deepcopy(simple_network)
    new_network.randomizeNodes(usableInput,usableOutput,epochs=1,rand=(random.random()-.5)/10)
    return [new_network,main(vectorRects,False,simple_network=new_network)]

def networkInputs(vectorRects,usableInput,usableOutput):
    # Used to know when to show the network running
    numIter = 1
    
    global scoreList
    
    # How many different neural networks we want at a given time
    length = 14
    simple_network = initializeNeuralNetwork(.5)
    simple_network.fit(usableInput,usableOutput, epochs=5000)
    main(vectorRects,True,simple_network)
    
    # This is when the third part of training start
    for i in range(length):
        scoreList.append([simple_network,main(vectorRects,False,simple_network=copy.deepcopy(simple_network))])

    # Sort the scorelist
    scoreList = sorted(scoreList,key=lambda l:l[1], reverse=True)

    # Run Mainloop
    while True:
        # Removes the bad networks from the list
        scoreList = scoreList[0:length]        

        # Print out current state of networks
        print("\n")
        for net in scoreList:
            print(net)

        pool = mp.Pool(processes=14)
        newScoreList = [pool.apply(runRandom, args=(scoreList[x][0],usableInput,usableOutput, vectorRects)) for x in range(0,length)]
        scoreList = scoreList + newScoreList
        
        # Sort the scorelist
        scoreList = sorted(scoreList,key=lambda l:l[1], reverse=True)
    
        if (numIter % 10 == 0):
            main(vectorRects,True,scoreList[0][0])
        numIter += 1

if __name__ == "__main__":
    # Initialize the board
    vectorRects = getVectorRects()
    
    # Run the main application to show the user the board setup
    main(vectorRects, True, usableInputReturn=False, numOfTicks=150)
    
    # Get usable input
    usableInput = main(vectorRects, False, usableInputReturn=True, numOfTicks=2000)[0:3, 1:]
    usableInput = np.reshape(usableInput, (3,int(usableInput.size/3),1))
    bigUsableInput = np.empty((1,2), np.float16)
    for goodInput,goodInput2 in zip(usableInput[0],usableInput[1]):
        bigUsableInput = np.append(bigUsableInput, [[goodInput[0], goodInput2[0]]], axis=0)
        
    # Get rid of first bad input
    bigUsableInput = bigUsableInput[1:]

    # Runs the network function
    networkInputs(vectorRects,bigUsableInput,usableInput[2])

