import numpy as np
from tkinter import *
import random
import math
import cProfile
import RaindropNeuralNetwork as RNN
from RaindropCanvas import *
import copy
import multiprocessing as mp
import time
import threading as th

# This variable determines how fast the goalpost moves
global goalPostChange
goalPostChange = 2

# Starting point for the goal
global goal_x,goal_x2
goal_x,goal_x2 = 250,350

# Change from user to generated
global generateDrops
generateDrops = True

# Default Neural Network
global default_network
default_network = RNN.RaindropNueralNetwork([2,3,3,1], alpha=.5)
default_network.W = [np.array([[-6.61760619, -2.11310018, -3.85155274,  1.58702452],[-1.1445116 , -0.56500305, -2.09672162,  1.7655994 ],[ 1.55662052, -1.71109883,  1.62429721, -1.10557543]]), 
       np.array([[ 3.47943931, -3.42554463, -1.04431427, -2.70289955],[ 0.52080882,  0.53738605,  0.97461208, -0.12834524],[ 3.07125882, -1.34274022,  0.07357841, -0.33383239],[-2.89892101, -0.20836237, -1.60680123, -1.49391823]]), 
       np.array([[-2.46899092],[ 2.69640724],[-0.03611722],[ 1.23841474]])]

# For when it is time to run the default network
global runningNetwork
runningNetwork = False

# Amount of ticks to keep the past score
global scoreTickNum
scoreTickNum = 800

# This gets the average tick amount to fall to the bottom
globalTickNum = 0
tickNumCount = 0

# Deletes old scores from the array
def updateScore(curtick,scoreNumList):
    global scoreTickNum
    badScoresLeft = True
    
    while badScoresLeft:
        if (len(scoreNumList) >= 1):
            if curtick - scoreNumList[0,1] > scoreTickNum:
                scoreNumList = np.delete(scoreNumList,0,0)
            else:
                badScoresLeft = False
        else:
            badScoresLeft = False
    
    return scoreNumList

def getGoalNum(goingRight):
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


def update(drops, score, curtick, goingRight, neural_network=None, usableInput=None, usableReturn=False):
    # Update the goal
    global vectorRects, generateDrops
    global goal_x, goal_x2
    global globalTickNum, tickNumCount
    global scoreNumList, scoreLabel
    
    scoreNumList = updateScore(curtick,scoreNumList)
    
    # Moves Goal
    goal_x,goal_x2,goingRight = moveGoal(goal_x,goal_x2,goingRight)
    if (VISUALIZE): 
        # This moves the goalpost on the screen
        global goal
        cvs.moveto(goal, goal_x, 600)
        
        # Updates the score on the screen 
        scoreLabel['text'] = str(math.trunc((np.sum(scoreNumList, axis=0)[0]*100))/100)
    
    # Gets right or left and converts it
    goalNum = getGoalNum(goingRight)

    # Gets x_coor from network
    if neural_network != None: 
        currentGoal = (goal_x+goal_x2)/2
        
        # Gets the goal Num
        newGoalNum = getGoalValue(goalNum,futureGoalNum(currentGoal, goalNum, int(globalTickNum)),currentGoal,futureGoal(currentGoal, goalNum, int(globalTickNum)))
        NN_x_coor = neural_network.predict(([(goal_x+goal_x2)/1200,newGoalNum]))[0][0]*600
    #print(f"Coor: {NN_x_coor}")

    # This adds more drops
    if (generateDrops or runningNetwork):
        if (VISUALIZE):
            drop,x_coor = addRandomDrop(False, VISUALIZE, cvs) # xcoor is the x coordinate at which the drop spawns
            #drops = np.append(drops,[[0.0,0.0,x_coor,0.0,drop]], axis=0)
            if neural_network != None:
                drops = np.append(drops,[[0.0,0.0,NN_x_coor,0.0,(goal_x+goal_x2),NN_x_coor,goalNum,drop]], axis=0) # Current X Velocity, Current Y Velocity, Current X Position, Current Y Position, goal_x at drop time (Used for neural network), original drop x (Used for neural network),goalNum, Drop Object
            else:
                drops = np.append(drops,[[0.0,0.0,x_coor,0.0,(goal_x+goal_x2),x_coor,goalNum,drop]], axis=0) # Current X Velocity, Current Y Velocity, Current X Position, Current Y Position, goal_x at drop time (Used for neural network), original drop x, goalNum, Drop Object
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
                dropScore = 1 - (abs((goal_x + goal_x2)/2 - drops[dropNum,2])/((goal_x2 - goal_x)*6))
                score += dropScore
                
                # Updates the current score to screen
                scoreNumList = np.append(scoreNumList,np.array([[dropScore,curtick]]),axis=0)
                    
                #print(f"Score: {score} \nScore Remover: {(abs((goal_x + goal_x2)/2 - drops[dropNum,2])/(goal_x2 - goal_x))}")
                diff = goal_x2-goal_x
                if usableReturn and (goal_x+diff/4 < drops[dropNum,2] < goal_x2+diff/4): 
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
    if usableReturn: return drops,usableInput,goingRight
    return drops,score,goingRight

def moveGoal(current_x1, current_x2, goingRight):
    # Changes direction if the post gets too close to the edges
    if current_x2 >= 600:
        goingRight = False
    elif current_x1 <= 0:
        goingRight = True
    
    # This maintains current direction
    if goingRight:
        return current_x1+goalPostChange, current_x2+goalPostChange, goingRight
    else:
        return current_x1-goalPostChange, current_x2-goalPostChange, goingRight
    

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
    global repeat, drops
    
    # Gets right or left and converts it # NEEDS TO BE FIXED
    goalNum = 1
    
    drop,x_coor = addRandomDrop(True, True, cvs)
    for i in range(4):
         drops = np.append(drops, [[0.0,0.0,x_coor,0.0,(goal_x+goal_x2),x_coor,goalNum,drop]], axis=0)
    repeat = root.after(4, addDropBlur, e)
    
    
def userScene(cvs):
    root.bind('<Motion>',callback)
    root.bind('<ButtonRelease-1>', stopRepeatDrop)
    root.bind("<Button-1>", addDropBlur)

def stopRepeatDrop(e):
    global repeat
    root.after_cancel(repeat)
    repeat = None

def changeToUser(canvas):
    global generateDrops,runningNetwork
    generateDrops = False
    runningNetwork = False
    userScene(canvas)

def changeToGenerated(canvas):
    global generateDrops,runningNetwork
    generateDrops = True
    runningNetwork = False
    generatedScene(canvas)
    
def changeToNeuralNetwork(canvas):
    global generateDrops,runningNetwork
    generateDrops = False
    runningNetwork = True
    userScene(canvas)
    generatedScene(canvas)
    
def respawnVectorRects():
    global vectorRectsObjects
    global cvs
    global vectorRects
    global drops
    
    # Gets right or left and converts it
    goalNum = 1
    
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
    drops = np.array([[0.0, 0.0, 0.0, 0.0, (goal_x+goal_x2), 0.0,goalNum, cvs.create_oval(0, 0, 2, 2, fill="#190482")], [0.0, 0.0, 50.0, 0.0, (goal_x+goal_x2), 50.0, goalNum, cvs.create_oval(50, 50, 52, 52, fill="#190482")]], np.float32)
    vectorRectsObjects = getVectorRectVisuals(cvs, vectorRects)
    
def initializeNeuralNetwork(alpha=0.5):
    return RNN.RaindropNueralNetwork([2,3,3,1], alpha=alpha)
  
def main(vectorRectsInput, visualize, simple_network=default_network, usableInputReturn=False, numOfTicks=scoreTickNum): 
    global vectorRects
    global drops
    vectorRects = vectorRectsInput
    #global drops
    score = 0
    
    # Starting direction of goal (True = right, False = left)
    goingRight = True
    
    # This is a list of drops that scored and the tick number that they scored on
    global scoreNumList
    scoreNumList = np.zeros((1,2))

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
        
        neuralNetworkButton = Button(root, text="NeuralNetwork", command=functools.partial(changeToNeuralNetwork, canvas=cvs))
        neuralNetworkButton.place(x=610, y=75)

        global scoreLabel
        scoreLabel = Label(root, text="0")
        scoreLabel.place(x=610, y=110)
        #scoreLabel.pack()
    
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
    curtick = 0
    while curtick<numOfTicks:
        # Updates the function
        if runningNetwork and usableInputReturn:
            drops,usableInput,goingRight = update(drops, score, curtick, goingRight, neural_network=simple_network, usableInput=usableInput, usableReturn=usableInputReturn)
        elif runningNetwork:
            drops,score,goingRight = update(drops, score, curtick, goingRight, neural_network=simple_network)
        elif usableInputReturn:
            drops,usableInput,goingRight = update(drops, score, curtick, goingRight, usableInput=usableInput, usableReturn=usableInputReturn)
        else:
            drops,score,goingRight = update(drops, score, curtick, goingRight)
        if(not VISUALIZE):
            curtick += 1
            continue
        
        if (not generateDrops):
            if (len(drops) < 2):
                time.sleep(.01)
            elif ((len(drops) < 15)):
                time.sleep(.007)
            elif (len(drops) < 35):
                time.sleep(.004)
        
        # Does the required visual updates
        root.update_idletasks()
        if curtick%4 == 0:
            root.update()
        curtick += 1
    if VISUALIZE: root.destroy()
    # This is for the section of the neural network in which we gather data
    if usableInputReturn: return usableInput
    return np.sum(scoreNumList, axis=0)[0]

def runRandom(simple_network, usableInput, usableOutput, vectorRects, epoch, networks, x):
    # Gets Scorelist
    global runningNetwork, generateDrops
    
    # For some reason need to reup this - not sure why
    runningNetwork = True
    generateDrops = False
    
    new_network = copy.deepcopy(simple_network)
    new_network.randomizeNodes(usableInput,usableOutput,epochs=1,rand=1/((epoch+1)/5))
    #networkScore = main(vectorRects,False,new_network)
    #scoreList.append([new_network,networkScore])
    networks[x] = new_network
    return networks

def networkInputs(vectorRects,usableInput,usableOutput):
    # Used to know when to show the network running
    epoch = 1
    
    # Makes sure to run network
    global runningNetwork
    runningNetwork = True
    
    global scoreList
    # Keeps track of all the scores of the neural networks
    scoreList = []
    
    # How many different neural networks we want at a given time
    length = 16
    simple_network = initializeNeuralNetwork(.4)
    print(simple_network.W);
    for i in range(10):
        simple_network.fit(usableInput,usableOutput, epochs=10)
        #print(f"Epoch - {(i+1)*10}, Score - {main(vectorRects,False,simple_network=simple_network)}")
        print(main(vectorRects,False,simple_network=simple_network))
    for i in range(50):
        simple_network.fit(usableInput,usableOutput, epochs=100)
        #print(f"Epoch - {(i+2)*100}, Score - {main(vectorRects,False,simple_network=simple_network)}")
        print(main(vectorRects,False,simple_network=simple_network))
    main(vectorRects,True,simple_network=simple_network)
    
    #printvector = np.array_repr(simple_network.W).replace('\n', '')
    print(simple_network.W)
    
    # This is when the third part of training start
    for i in range(length):
        scoreList.append([simple_network,main(vectorRects,False,simple_network=copy.deepcopy(simple_network))])

    # Sort the scorelist
    scoreList = sorted(scoreList,key=lambda l:l[1], reverse=True)
    
    # Adds some epochs
    adjustedScore = scoreList[0][1] - 380
    
    if (adjustedScore > 0):
        epoch += int(adjustedScore/3)
        
    maxEpochNum = epoch+200

    # Run Mainloop
    while (epoch <= maxEpochNum):
        # Fix does not work
        pool = mp.Pool(processes=4)
        networks = [None] * length
        for x in range(0,length):
            networks = pool.apply(runRandom, args=(scoreList[x][0],usableInput,usableOutput,vectorRects,epoch,networks,x))

        for i,value in enumerate(networks):
            scoreList.append([value,main(vectorRects,False,value)])   
            
        print(scoreList)
            
        # Sort the scorelist
        scoreList = sorted(scoreList,key=lambda l:l[1], reverse=True)
        
        # Removes the bad networks from the list
        scoreList = scoreList[0:length]  
        
        # Print out current state of networks
        print(f"\nEpoch Number: {epoch}")
        print("\n")
        for net in scoreList:
            print(net)  
    
        if (epoch % 10 == 0):
            print(main(vectorRects,False,scoreList[0][0]))
            print(main(vectorRects,True,scoreList[0][0]))
        epoch += 1
        
    print(scoreList[0][0].W)
        
    return scoreList[0][0]

def trainNewNetwork(vectorRects):
    # Get usable input
    usableInput = main(vectorRects, False, usableInputReturn=True, numOfTicks=2500)[0:3, 1:]

    # Make sure that no generated 
    generateDrops = False
    
    usableInput = np.reshape(usableInput, (3,int(usableInput.size/3),1))
    bigUsableInput = np.empty((1,2), np.float16)
    for goodInput,goodInput2 in zip(usableInput[0],usableInput[1]):
        bigUsableInput = np.append(bigUsableInput, [[goodInput[0], goodInput2[0]]], axis=0)
        
    # Get rid of first bad input
    bigUsableInput = bigUsableInput[1:]

    # Runs the network function
    new_network = networkInputs(vectorRects,bigUsableInput,usableInput[2])
    
    return new_network
    

if __name__ == "__main__":    
    # Initialize the board with set board
    # The data present is the default array for the vector rects
    vectorRects = np.array([[[-0.00786831, -0.09884162],        [-0.00873844, -0.10053781],        [ 0.01332159, -0.05923093],        [-0.0117655 , -0.06682798],        [-0.00694016, -0.06501867],        [-0.04253722, -0.13260111],        [ 0.02559236, -0.06737399],        [-0.02479084, -0.10036221],        [-0.02165457, -0.1272367 ],        [ 0.01598465, -0.08073852],        [ 0.04490769,  0.02214127],        [-0.00498738,  0.02775725],        [ 0.03546589,  0.11223152],        [-0.01297482,  0.10227705],        [-0.0078466 ,  0.10231937]],       [[-0.01663717,  0.08572452],        [ 0.03094651,  0.09864197],        [-0.052846  , -0.06803727],        [ 0.00268107, -0.05053785],        [-0.0337498 , -0.08968223],        [-0.01527273, -0.07523458],        [-0.00780774, -0.10471047],        [-0.05663672, -0.17951962],        [-0.00218093, -0.13356286],        [ 0.01876004, -0.12243037],        [ 0.03034475, -0.10363717],        [-0.00631324, -0.10925583],        [-0.00883816, -0.11593931],        [-0.05133557, -0.14734249],        [-0.06079391, -0.1741985 ]],       [[ 0.01338048, -0.14296485],        [ 0.0281988 , -0.02285494],        [-0.05159537, -0.04640648],        [-0.07245473, -0.15849487],        [-0.01217295, -0.10900791],        [-0.04364195, -0.15374716],        [-0.02343229, -0.13617283],        [ 0.00539762, -0.05175057],        [ 0.04215109,  0.01193801],        [ 0.0669399 ,  0.06720905],        [-0.01074327, -0.04728954],        [ 0.04907575,  0.02898885],        [ 0.03231163,  0.08625402],        [-0.06760605, -0.01956932],        [-0.09696806, -0.13308923]],       [[ 0.04660281, -0.0768227 ],        [-0.03317572, -0.17400081],        [-0.06642057, -0.18455628],        [-0.09779304, -0.22926626],        [ 0.01135684, -0.12193387],        [ 0.02603075, -0.00521006],        [-0.00388566, -0.01461298],        [ 0.04708269,  0.02738319],        [ 0.02842742, -0.01734312],        [ 0.10050675,  0.10210208],        [ 0.01497441,  0.08105394],        [-0.01382503, -0.04538162],        [-0.02644557, -0.10120712],        [-0.0975273 , -0.13565782],        [-0.09215241, -0.10896118]],       [[-0.00479146, -0.163407  ],        [ 0.0253904 , -0.04439035],        [-0.03618846, -0.02440714],        [-0.08508667, -0.1089972 ],        [-0.01250054, -0.14902079],        [-0.01085511, -0.19320293],        [ 0.03898596, -0.09017339],        [ 0.08358324,  0.03404913],        [ 0.00803207, -0.04786833],        [ 0.04470496, -0.07784054],        [ 0.02619748, -0.02817319],        [-0.04388966, -0.06182243],        [-0.05210791, -0.08111168],        [-0.11546668, -0.15586573],        [-0.07887836, -0.12096458]],       [[ 0.02297198, -0.084148  ],        [ 0.0017293 , -0.11293015],        [-0.04307983, -0.11182319],        [-0.0709125 , -0.11230408],        [-0.00578531, -0.06681161],        [ 0.02459761, -0.04323292],        [ 0.08340186,  0.00156434],        [ 0.01328029, -0.10923328],        [ 0.03302195, -0.06809647],        [-0.01133952, -0.1469513 ],        [ 0.00749455, -0.14023173],        [-0.0743747 , -0.19411014],        [-0.11571494, -0.25316456],        [-0.06017845, -0.13906407],        [-0.11982549, -0.20284906]],       [[-0.02495399, -0.25316456],        [ 0.00969164, -0.17749146],        [-0.01035443, -0.10223024],        [-0.04713789, -0.09989101],        [-0.06760138, -0.2086511 ],        [ 0.0153367 , -0.19757824],        [ 0.01537532, -0.22767222],        [ 0.01348007, -0.19967307],        [ 0.01416783, -0.18142661],        [-0.03371011, -0.21970072],        [-0.05704474, -0.25316456],        [-0.03429887, -0.15978242],        [-0.11925381, -0.21969745],        [-0.06995519, -0.19756849],        [-0.06743871, -0.11243849]],       [[-0.07246798, -0.22492453],        [ 0.02697522, -0.1753079 ],        [-0.00058374, -0.14948817],        [-0.10078067, -0.2473886 ],        [-0.01001685, -0.13756439],        [ 0.04822209, -0.05763893],        [-0.0321165 , -0.19051977],        [ 0.03452203, -0.16157283],        [-0.04167984, -0.25316456],        [-0.04676106, -0.25316456],        [-0.01517371, -0.18211566],        [-0.06296325, -0.21117423],        [-0.04043869, -0.10189882],        [-0.1132566 , -0.20022956],        [-0.08317954, -0.22868873]],       [[-0.09047168, -0.25316456],        [ 0.04544719, -0.15258324],        [-0.00043719, -0.12679991],        [-0.04310329, -0.10564141],        [-0.01619914, -0.14497355],        [-0.02378179, -0.22176127],        [ 0.04972726, -0.0790629 ],        [-0.04858321, -0.21056323],        [-0.02021956, -0.18582946],        [-0.00980249, -0.14983365],        [-0.08648561, -0.25316456],        [-0.06863409, -0.25316456],        [-0.10473825, -0.25316456],        [-0.12987013, -0.25316456],        [-0.11479698, -0.25316456]],       [[-0.07722109, -0.25316456],        [-0.00068844, -0.25316456],        [-0.04253451, -0.25316456],        [-0.07596944, -0.25316456],        [-0.00171786, -0.16626711],        [ 0.0420255 , -0.06764845],        [-0.02307064, -0.17876302],        [-0.06612562, -0.23975751],        [ 0.02500379, -0.13506302],        [-0.00401559, -0.10012484],        [-0.02175234, -0.03043862],        [-0.09764664, -0.12859794],        [-0.11593972, -0.1919321 ],        [-0.12987013, -0.25316456],        [-0.1066233 , -0.22858895]],       [[-0.01336603, -0.1479915 ],        [ 0.00450858, -0.11634759],        [-0.03172407, -0.13924082],        [-0.05105874, -0.17189692],        [ 0.04726596, -0.08679543],        [ 0.02810422, -0.08908711],        [ 0.02846867,  0.00792236],        [ 0.01752514,  0.05663216],        [ 0.03819337,  0.04840053],        [ 0.05627551,  0.10697152],        [-0.05051145,  0.01576477],        [-0.04138172,  0.08633699],        [-0.05897564,  0.13223192],        [-0.09654752,  0.08537663],        [-0.12987013, -0.0511715 ]],       [[ 0.02977946,  0.02268662],        [-0.00301556, -0.01820855],        [-0.04334812, -0.07721896],        [-0.04268627, -0.13126723],        [ 0.03213175, -0.12066831],        [ 0.07294293, -0.03194944],        [ 0.08270706,  0.06198577],        [ 0.07929196,  0.15248576],        [ 0.01001589,  0.070511  ],        [ 0.03016238,  0.0684256 ],        [ 0.00863807,  0.14371127],        [-0.00129505,  0.19606318],        [-0.1147488 ,  0.06391941],        [-0.12987013, -0.02478679],        [-0.0700334 ,  0.10515267]],       [[-0.03515653, -0.00262862],        [-0.03538249, -0.05493858],        [ 0.02166403,  0.06443483],        [-0.02704366,  0.02041112],        [ 0.05133967,  0.02830946],        [ 0.015363  , -0.11399693],        [ 0.02537898, -0.19577989],        [ 0.10987892, -0.03116234],        [ 0.05045873,  0.06207217],        [ 0.04760581,  0.11701956],        [-0.00556651,  0.06990193],        [-0.05771787,  0.02748533],        [-0.11625074,  0.03057423],        [-0.12987013,  0.00356267],        [-0.12987013, -0.10215539]],       [[ 0.01866028,  0.0028829 ],        [ 0.0248908 ,  0.09338276],        [ 0.02178227,  0.09684246],        [-0.00101482,  0.08174443],        [ 0.02193109,  0.05704292],        [-0.00993388, -0.02109749],        [-0.00507244, -0.14078107],        [ 0.04026055, -0.22890688],        [ 0.10897622, -0.08173611],        [ 0.08445533,  0.0546871 ],        [ 0.04207668,  0.16544193],        [-0.10406789,  0.04981599],        [-0.06719095,  0.11898532],        [-0.0870716 ,  0.1703782 ],        [-0.12471739,  0.13135938]],       [[-0.00394848,  0.08371439],        [-0.01459107, -0.0128867 ],        [ 0.0412368 ,  0.06501583],        [ 0.04474057,  0.14330478],        [-0.01606931,  0.07540536],        [ 0.01395749,  0.1003065 ],        [-0.02430027,  0.00918691],        [ 0.04315859, -0.03963138],        [ 0.10464744, -0.06243169],        [ 0.02624662, -0.17366794],        [ 0.05147594, -0.04220984],        [-0.02612703,  0.08971408],        [-0.12987013, -0.02128438],        [-0.06262556,  0.07549635],        [-0.0752822 ,  0.17818613]]])
    
    # # Run the main application to show the user the board setup
    # main(vectorRects, True, usableInputReturn=False, numOfTicks=150)
    
    # # Get usable input and MAKE SURE DIRECTION INPUT FOR NEURAL NETWORK WORKS CORRECTLY
    usableInput = main(vectorRects, False, usableInputReturn=True, numOfTicks=2500)[0:3, 1:]
    
    # Train a new neural network
    #new_network = trainNewNetwork(vectorRects)
    
    # Runs the trained network indefinetly

    # Run with new network
    # main(vectorRects,True,simple_network=new_network,numOfTicks=1000000)

    # Run with default network
    main(vectorRects,True,simple_network=default_network,numOfTicks=1000000)

