from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

# Import np arrays for cell images and labels
cells = np.load( 'data/cells.npy')
labels = np.load( 'data/labels.npy')

# generate train/test splits on data
X_train,X_test,y_train,y_test=train_test_split(cells,labels,test_size=0.2,random_state=1)
print(len(X_train),len(X_test))
print(len(y_train),len(y_test))
print(X_train[100].shape)
#Doing One hot encoding as classifier has multiple classes
y_train=keras.utils.to_categorical(y_train,2)
y_test=keras.utils.to_categorical(y_test,2)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
# Set up params for convolutional neural network 
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(46, 40, 3))) 
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))   
model.add(Dense(2, activation='softmax'))
print("\n********************Model Summary************************\n")
model.summary() 
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=["accuracy"])
model.fit(X_train,y_train, epochs=10, batch_size=52, shuffle=True, validation_data=(X_test,y_test))
# print fitted CNN model accuracy on unseen data
accuracy = model.evaluate(X_test, y_test, verbose=1)
print('\n', 'Test_Accuracy:-', accuracy[1])

from sklearn.metrics import classification_report
# predict probabilities for test set
y_pred = model.predict(X_test, verbose=0)
# predict crisp classes for test set
classes = model.predict_classes(X_test, verbose=0)
print("\n****************Classification Report********************\n")
print(classification_report(y_test, y_pred.round()))


from keras.models import load_model
model.save('/users/sath/Documents/Projects/malaria-detection/models/cells_cnn.h5')