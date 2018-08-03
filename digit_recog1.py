import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
cam=cv.VideoCapture(0)
img = cv.imread('digits.png')
print(img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)
# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()
# Initiate kNN, train the data, then test it with test data for k=1
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print( accuracy )
# save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)
# Now load the data
with np.load('knn_data.npz') as data:
    print( data.files )
    train = data['train']
    train_labels = data['train_labels']



names = {
    0: 'zero',
    1: 'one',
    2:'two',
    3:'three',
    4:'four',
    5:'five',
    6:'six',
    7:'seven',
    8:'eight',
    9:'nine'
}

font = cv.FONT_HERSHEY_SIMPLEX

def distance(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum())


def knn(x, train, targets, k=5):
    m = train.shape[0]
    dist = []
    for ix in range(m):
        # compute distance from each point and store in dist
        dist.append(distance(x, train[ix]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    print("Index...", indx)
    sorted_labels = train_labels[indx][:k]
    print("Sorted...", sorted_labels)
    counts = np.unique(sorted_labels, return_counts=True)
    print("Count...", counts)
    return counts[0][np.argmax(counts[1])]

while True:
    ret, frame = cam.read()
    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        print(gray)



        for (x, y, w, h) in  :
            digit_component = frame[y:y + h, x:x + w, :]
            fc = cv.resize(digit_component, (28, 28))

            lab = knn(fc.flatten(), data,labels)
            text = names[int(lab)]
        cv.putText(frame, text, (x, y), font, 1, (255, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imshow('digit recog', frame)
        k = cv.waitKey(33) & 0xFF
        if k == 27:
            break
    else:
        print('Error')

cam.release()
cv.destroyAllWindows()