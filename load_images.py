import numpy as np
import os
from PIL import Image
from resizeimage import resizeimage

images=[]
labels=[]

# define function to load segmented cell images and resize to liking
def load_images_from_folder_a(folder,id):
    
    for filename in os.listdir(folder):
        if filename!="Thumbs.db":
            img1 = Image.open(os.path.join(folder,filename))
            new1 = resizeimage.resize_contain(img1, [40, 46, 3])
            new1 = np.array(new1, dtype='uint8')
            images.append(new1)
            if id==1:
                labels.append(1)
            else:
                labels.append(0)

# apply function to load in cell images and generate labels.
load_images_from_folder_a("/users/sath/Documents/Projects/malaria-detection/cell_images/Parasitized",1)
load_images_from_folder_a("/users/sath/Documents/Projects/malaria-detection/cell_images/Uninfected/",2)

print(len(images))
print(len(labels))

# save off cell images and labels as numpy objects to load in for modelling.
cells = np.array(images)
cells = cells[...,:3] # png images have a fourth invisible layer so we select the first three to get RGB
cells = cells.astype('float32') / 255 
labels = labels

np.save( 'data/cells.npy' , cells )
np.save( 'data/labels.npy' , labels )

print('Cells : {} | labels : {}'.format(cells.shape , len(labels)))

