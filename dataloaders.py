import os 

import cv2 

class VideoLoader:
    def __init__(self, path):
        super().__init__()

        self.path = path 
        

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.path)
        self.max = self.cap.get(7)
        self.curr_frame = 0
        return self 


    def __next__(self):
        if self.curr_frame < self.max:
            self.cap.set(1, self.curr_frame)
        ret, img = self.cap.read()
        self.curr_frame += 1

        if ret:
            return img 
        else:
            self.cap.release()
            raise StopIteration

    def __exit__(self):
        self.cap.release()

    def peek(self):
        if self.curr_frame + 1 >= self.max:
            return False, None

        self.cap.set(1, self.curr_frame+1)
        ret, img = self.cap.read()
        return ret, img
        



class ImageLoader:
    def __init__(self, path):
        super().__init__()

        self.path = path 
        self.images = sorted([image for image in os.listdir(path) if image[-3:]=='png' or image[-3:] == 'jpg'])
    

    def __iter__(self):
        self.index = 0
        return self 


    def __next__(self):
        img = cv2.imread(os.path.join(self.path, self.images[self.index]))
        self.index += 1
        if self.index >= len(self.images):
            raise StopIteration
        return img

    def __exit__(self):
        return 

    def peek(self):
        if self.index + 1 >= len(self.images):
            return False, None 

        img = cv2.imread(os.path.join(self.path, self.images[self.index+1]))
        return True, img
