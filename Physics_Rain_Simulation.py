from numpy import delete, integer
import numpy as np
from tkinter import *
import time
import random
# import pyautogui

def update(cvs, wind):
        global drops 
        #print(drops)

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
                currentRect = + vectorRects[round((dropCoor[0]/30)-1),round((dropCoor[1]/30)-1)]
                #print("x vector", currentRect[0])
                drops[dropNum, 2] += .25 + currentRect[0]
                drops[dropNum, 1] += .0 + currentRect[1]
                #print(drops[dropNum])
                cvs.tick(drops[dropNum])
                dropNum += 1


#def getColor(xcomp, ycomp):
    

class RaindropCanvas(Canvas):
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dx = 0
        self.dy = 0
        
    def addRandomDrop(self, dropsList):
        x_coor = random.randint(0,596)
        size = random.randint(1,2)
        drop = self.create_oval(x_coor, 0, x_coor+size, size, fill="#190482", )
        global drops
        drops = np.append(dropsList, [[float(drop),0.0,0.0]], axis=0)
    
    # This function initializes the vector squares
    def getVectorRects(self):
        global vectorRects
        vectorRects = np.zeros((20,20,2), np.float32)

        for xDirInt in range(len(vectorRects)):
            for yDirInt in range(len(vectorRects[xDirInt])):
                vectorRects[xDirInt, yDirInt] = [(random.random()-.5)/2, random.random()-0.5] #Gives random vector components to each square
                self.create_rectangle((xDirInt-1)*30, (yDirInt-1)*30, ((xDirInt)*30)+30, ((yDirInt)*30)+30, outline="#ffffff") #This Creates a square every 30px on the screen
        
      
    def tick(self, droplist):
        # Just moves the drop
        self.move(int(droplist[0]), droplist[1], droplist[2]) # droplist[0] = drop, droplist[1] = dx, droplist[2] = dy
 
    def change_heading(self, dx, dy):
        self.dx = dx
        self.dy = dy
    
    def getcoor(self, drop):
        return self.coords(drop)
  

if __name__ == "__main__":
    
    global wind
    wind = 0.0

    root = Tk()
    root.geometry('600x600')
    #root.attributes('-alpha', 0.1)
    
    cvs = RaindropCanvas(root)
    cvs.pack(fill="both", expand=True)
    cvs.getVectorRects()

    drops = np.array([[cvs.create_oval(0, 0, 2, 2, fill="#190482"), 0, 0], [cvs.create_oval(50, 50, 52, 52, fill="#190482"), 0, 0]], np.float32)

    while True:
        update(cvs, wind)
        root.update_idletasks()
        root.update()

