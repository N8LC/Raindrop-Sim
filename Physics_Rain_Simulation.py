from numpy import integer
import numpy as np
from tkinter import *
import time
import random
# import pyautogui

class RaindropCanvas(Canvas):
    
    global drops
    drops = []
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dx = 0
        self.dy = 0
 
        self.dt = 25
        
    def addRandomDrop(self):
        x_coor = random.randint(0,596)
        drop = self.create_oval(x_coor, 0, x_coor+10, 10, fill="#190482")
        drops.append([drop,0,0])
        
      
    def tick(self, droplist):
        
        #for drop in drops:
        self.move(droplist[0], droplist[1], droplist[2]) # droplist[0] = drop, droplist[1] = dx, droplist[2] = dy
        #self.after(self.dt, self.tick)
 
    def change_heading(self, dx, dy):
        self.dx = dx
        self.dy = dy
  

if __name__ == "__main__":
    
    global wind
    wind = 0.0

    def update(cvs, wind):
        cvs.addRandomDrop()
        cvs.addRandomDrop()
        #print(drops)
        #wind += (random.random() - .5)
        for drop in drops:
            #wind += (random.random() - .5)
            if (drop[2] > 19):
                cvs.delete(drop[0])
                drops.remove(drop)
            drop[2] += .28
            #drop[1] += (random.random() - .5)#wind
            cvs.tick(drop)

    root = Tk()
    root.geometry('600x600')
    
    cvs = RaindropCanvas(root)
    cvs.pack(fill="both", expand=True)

    while True:
        update(cvs, wind)
        root.update_idletasks()
        root.update()
        time.sleep(.01);

