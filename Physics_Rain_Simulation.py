from os import startfile
from numpy import delete, integer
import numpy as np
from tkinter import *
import time
import random
import math
### Modules
from colorsys import hls_to_rgb
import cProfile
import re

import numpy
# import pyautogui

xNumOfDivs = 2.4
yNumOfDivs = 1.75 
minCumulitiveFactor = 3.5
maxColorVal = 1/math.sqrt((.5/yNumOfDivs) + (.5/xNumOfDivs)) # This just adjusts the formula so that the max color value is correctly calculated

def update(cvs, wind):
        global drops 

        for i in range(4):
            cvs.addRandomDrop(drops)
            
        dropNum = 0
        while dropNum < len(drops) - 1:
            dropCoor = cvs.getcoor(int(drops[dropNum, 0])) #This gets the drop object from the canvas and then gets its coordinate Reminder list of: [dropobject, dx, dy]
            if ((dropCoor[0] < 0 or dropCoor[0] > 600) or (dropCoor[1] < 0 or dropCoor[1] > 600)):
                cvs.delete(int(drops[dropNum, 0]))
                drops = np.delete(drops, dropNum, 0)
            else:
                global vectorRects
                currentRect = vectorRects[math.floor((dropCoor[0]/(600/size))),math.floor((dropCoor[1]/(600/size)))]
                drops[dropNum, 2] += .2 - currentRect[1] # Positive is down in tkinter so to simulate pyhsics we subtract instead of add
                drops[dropNum, 1] += .0 + currentRect[0]
                cvs.tick(drops[dropNum])
                dropNum += 1

def getColor(xcomp, ycomp):
    totalVector = math.sqrt(xcomp**2 + ycomp**2)

    vectorAngle = getVectorAngle(xcomp, ycomp)
    
    rgb = hls_to_rgb(vectorAngle/360, .75, totalVector*maxColorVal) # This makes sure that no matter what the max and min values are they show up as intended
    
    return '#{:02x}{:02x}{:02x}'.format(abs(int(rgb[0]*255)), abs(int(rgb[1]*255)), abs(int(rgb[2]*255)))

def getVectorAngle(xcomp, ycomp):
    vectorAngle = np.arctan2([abs(ycomp)], [abs(xcomp)])[0] * 180 / np.pi # This gets the angle for the colors

    if (xcomp < 0 and ycomp < 0):
        vectorAngle += 180
    elif (xcomp < 0):
        vectorAngle = 180 - vectorAngle
    elif (ycomp < 0):
        vectorAngle = 360 - vectorAngle
    
    print(vectorAngle, xcomp, ycomp)
    
    return vectorAngle
    

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
    

class RaindropCanvas(Canvas):
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dx = 0
        self.dy = 0
        
    def addRandomDrop(self, dropsList):
        x_coor = random.randint(0,596)
        size = random.random()*4.3 # This changes the size of the drop 4.3
        drop = self.create_oval(x_coor, 1, x_coor+size, size, fill="#190482", outline="#190482") # blue color #190482
        global drops
        drops = np.append(dropsList, [[float(drop),0.0,0.0]], axis=0)
    
    # This function initializes the vector squares
    def getVectorRects(self):
        global size
        size = 16
        
        baseVector = [(random.random()-.5)/xNumOfDivs, (random.random()-.5)/yNumOfDivs]

        global vectorRects
        vectorRects = np.zeros((size,size,2), np.float32)

        for xDirInt in range(len(vectorRects)):
            for yDirInt in range(len(vectorRects[xDirInt])):
                if (xDirInt == 0 and yDirInt == 0):                    
                    vectorRects[xDirInt, yDirInt] = baseVector #Gives random vector components to each square
                elif (yDirInt == 0):
                    vectorRects[xDirInt, yDirInt] = [vectorRects[xDirInt-1, yDirInt, 0] + (random.random()-.5)/(xNumOfDivs*minCumulitiveFactor), vectorRects[xDirInt-1, yDirInt+size-1, 1] + (random.random()-0.5)/(yNumOfDivs*minCumulitiveFactor)] #Gives random vector components to each square INTENTIONAL BUG
                    vectorRects[xDirInt, yDirInt] = doesNotExceedBounds(vectorRects[xDirInt, yDirInt, 0], vectorRects[xDirInt, yDirInt, 1])
                elif (xDirInt == 0):
                    vectorRects[xDirInt, yDirInt] = [baseVector[0] + (random.random()-.5)/(xNumOfDivs*minCumulitiveFactor), vectorRects[xDirInt, yDirInt-1, 1] + (random.random()-0.5)/(yNumOfDivs*minCumulitiveFactor)] #Gives random vector components to each square INTENTIONAL BUG
                    vectorRects[xDirInt, yDirInt] = doesNotExceedBounds(vectorRects[xDirInt, yDirInt, 0], vectorRects[xDirInt, yDirInt, 1])
                else:
                    vectorRects[xDirInt, yDirInt] = [vectorRects[xDirInt-1, yDirInt, 0] + (random.random()-.5)/(xNumOfDivs*minCumulitiveFactor), vectorRects[xDirInt, yDirInt-1, 1] + (random.random()-0.5)/(yNumOfDivs*minCumulitiveFactor)] #Gives random vector components to each square INTENTIONAL BUG
                    vectorRects[xDirInt, yDirInt] = doesNotExceedBounds(vectorRects[xDirInt, yDirInt, 0], vectorRects[xDirInt, yDirInt, 1])
                
                # Create the Square
                coordinates = [(xDirInt)*(600/size), (yDirInt)*(600/size), ((xDirInt)*(600/size))+(600/size), ((yDirInt)*(600/size))+(600/size)]
                self.create_rectangle(coordinates[0], coordinates[1], coordinates[2], coordinates[3], outline="#ffffff", fill=getColor(vectorRects[xDirInt, yDirInt, 0],vectorRects[xDirInt, yDirInt, 1])) #This Creates a square every (600/size)px on the screen
                self.create_text((coordinates[2] + coordinates[0])/2,(coordinates[3] + coordinates[1])/2, text=str(int(getVectorAngle(vectorRects[xDirInt, yDirInt, 0], vectorRects[xDirInt, yDirInt, 1]))), fill="white")
                
       
      
    def tick(self, droplist):
        # Just moves the drop
        self.move(int(droplist[0]), droplist[1], droplist[2]) # droplist[0] = drop, droplist[1] = dx, droplist[2] = dy
 
    def change_heading(self, dx, dy):
        self.dx = dx
        self.dy = dy
    
    def getcoor(self, drop):
        return self.coords(drop)
  
def main():
    global wind
    wind = 0.0

    root = Tk()
    root.geometry('600x600')
    #root.attributes('-alpha', 0.1)
    
    cvs = RaindropCanvas(root)
    cvs.pack(fill="both", expand=True)
    cvs.getVectorRects()

    global drops
    drops = np.array([[cvs.create_oval(0, 0, 2, 2, fill="#190482"), 0, 0], [cvs.create_oval(50, 50, 52, 52, fill="#190482"), 0, 0]], np.float32)

    i = 0
    root.update()

    while True:
        update(cvs, wind)
        root.update_idletasks()
        if i%20 == 0:
            root.update()
        i += 1

if __name__ == "__main__":
    main()

