import numpy
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

img_pred = cv2.imread ( 'test2.png' ,   0 )
# forces the image to have the input dimensions equal to those used in the training data (28x28)
if img_pred.shape != [ 28 , 28 ]:
    img2 = cv2.resize ( img_pred, ( 28 , 28 ) )
    img_pred = img2.reshape ( 28 , 28 , - 1 ) ;
else :
    img_pred = img_pred.reshape ( 28 , 28 , - 1 ) ;
    
# here also we inform the value for the depth = 1, number of rows and columns, which correspond 28x28 of the image.
img_pred = img_pred.reshape ( 1 , 1 , 28 , 28 )
pred = model.predict_classes ( img_pred )
pred_proba = model.predict_proba ( img_pred )
pred_proba = "% .2f %%" % (pred_proba [0] [pred] * 100) 
print ( pred [ 0 ] , "with probability of" , pred_proba )

















