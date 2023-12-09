from tkinter import *
import numpy as np
import random
import functools
from colorsys import hls_to_rgb
import math

global drops

xNumOfDivs = 3.85 # Increasing this decreases the amount of variation in the x vector components
yNumOfDivs = 1.975 # Increasing this decreases the amount of variation in the y vector components
minCumulitiveFactor = 2.23
maxColorVal = (1/math.sqrt((.5/yNumOfDivs)**2 + (.5/xNumOfDivs)**2))/1.65 # This just adjusts the formula so that the max color value is correctly calculated

global size
size = 15


global x0
x0 = 0

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
    
    return vectorAngle

def InitializeCanvas(vectorRects):
    global root
    root = Tk()
    root.geometry('750x620')    

    # This gets the canvas
    cvs = RaindropCanvas(root)
    cvs.pack(fill="both", expand=True)
    
    # This gets the tkinter representation of the vector rectangles
    vectorRectsObjects = getVectorRectVisuals(cvs, vectorRects)
    
    return cvs,vectorRectsObjects,root

def generatedScene(cvs):
    root.unbind('<Motion>')
    root.unbind('<ButtonRelease-1>')
    root.unbind("<Button-1>")
    
# The following three functions control the user inputs
def callback(e):
   global x0
   x0 = e.x
    
def addRandomDrop(player, VISUALIZE, *canvas):
        x_coor = 0
        if player:
            global x0
            x_coor = x0
        else:
            x_coor = random.randint(0,596)
        size = (random.random()+.4)*4.3 # This changes the size of the drop 4.3, the addition gives a bottom line for the size
        
        # No need to create a drop object if we are not showing it
        if (not VISUALIZE):
            return x_coor
        
        # For if we are showing it
        for s in canvas:
            drop = s.create_oval(x_coor, 1, x_coor+size, 1+size, fill="#190482", outline="#190482") # blue color #190482
        return drop, x_coor
            
def getVectorRectVisuals(canvas, vectorRects):
    # This initializes vectorRectObjects
    vectorRectsObjects = np.zeros((size**2, 2), np.int32)
    
    for xDirInt in range(len(vectorRects)):
        for yDirInt in range(len(vectorRects[xDirInt])):
            # Create the Square
            coordinates = [(xDirInt)*(600/size), (yDirInt)*(600/size), ((xDirInt)*(600/size))+(600/size), ((yDirInt)*(600/size))+(600/size)]
            squareColor = getColor(vectorRects[xDirInt, yDirInt, 0],vectorRects[xDirInt, yDirInt, 1])
                
            # Change Color Here
            vectorObject = canvas.create_rectangle(coordinates[0], coordinates[1], coordinates[2], coordinates[3], outline=squareColor, fill=squareColor) #This Creates a square every (600/size)px on the screen
            #vectorText = self.create_text((coordinates[2] + coordinates[0])/2,(coordinates[3] + coordinates[1])/2, text=str(int(getVectorAngle(vectorRects[xDirInt, yDirInt, 0], vectorRects[xDirInt, yDirInt, 1]))), fill="white")
                
            vectorRectsObjects[(size*xDirInt)+yDirInt] = [vectorObject, 0]#vectorText]
            
    return vectorRectsObjects

class RaindropCanvas(Canvas):
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dx = 0
        self.dy = 0
      
    def tick(self, droplist):
        # Just moves the drop
        self.moveto(int(droplist[7]), droplist[2], droplist[3]) # droplist[4] = drop, droplist[0] = dx, droplist[1] = dy
 
    def change_heading(self, dx, dy):
        self.dx = dx
        self.dy = dy
