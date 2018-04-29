import numpy as np
import pandas as pd
import gzip
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#learning_rate = 0.1
#learning_rate = 0.01
learning_rate = 0.001


f = gzip.open("test-images.gz", 'rb')
test_images = np.frombuffer(f.read(), np.uint8, offset=16)
test_images = test_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(test_images.shape[0],784)

f = gzip.open("test-labels.gz", 'rb')
test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
test_labels = test_labels.reshape(10000,1)



test_predicted = test_labels.transpose().astype(int).reshape(10000,)

#convert lables with index 1
test_labels_final = np.zeros([10000,10],dtype=int)

for row in range(len(test_images)):
    index = test_labels[row].astype(int)
    test_labels_final[row][index[0]] += 1

#preprocessing
test_images_matrix = np.copy(test_images)
test_images_matrix = test_images_matrix / np.float32(255)
bias_col = np.ones((10000,1))
test_images_matrix = np.append(test_images_matrix, bias_col,axis = 1)

test_y = np.zeros((10000,1))
test_actual = np.zeros((1,10000))

y_current = np.zeros((10000,10))
accuracy_test = np.zeros((70))

##################################################################  TRAIN  ###########################################################

f = gzip.open("train-images.gz", 'rb')
train_images = np.frombuffer(f.read(), np.uint8, offset=16)
train_images = train_images.reshape(-1, 1, 28, 28)
train_images = train_images.reshape(train_images.shape[0],784)

f = gzip.open("train-labels.gz", 'rb')
train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
train_labels = train_labels.reshape(60000,1)


train_predicted = train_labels.transpose().astype(int).reshape(60000,)


#convert lables with index 1
train_labels_final = np.zeros([60000,10],dtype=int)

for row in range(len(train_images)):
    index = train_labels[row].astype(int)
    train_labels_final[row][index[0]] += 1

#preprocessing
train_images_matrix = np.copy(train_images)
train_images_matrix = train_images_matrix / np.float32(255)
bias_col = np.ones((60000,1))
train_images_matrix = np.append(train_images_matrix, bias_col,axis = 1)


# creating weights
weights = np.random.uniform(-0.05, 0.05, size=(10,785))

train_y = np.zeros((60000,1))
train_actual = np.zeros((1,60000))
y_current = np.zeros((60000,10))
accuracy_train = np.zeros((70))

for i in range(70):
#################################################### TRAIN  CALCULATION #########################################
    for row in range(len(train_images_matrix)):
        y_final_train = np.zeros([1,10],dtype=int)
        y_train = np.matmul(train_images_matrix[row:row+1],np.transpose(weights))
        maxindex_train = np.argmax(y_train, axis = 1)
        y_final_train[0,maxindex_train[0]] = 1
        delta_train = train_labels_final[row:row+1] - y_final_train
        #print "train_labels_final = {}, train_images_matrix = {}".format(train_labels_final[row:row+1] ,(train_images_matrix[row:row+1]))
        index_y_train = y_final_train.argmax(axis = 1)
        train_y[row] = index_y_train
        if(delta_train.any(axis=1)):
        # print("update needed")
            weights = weights + 0.001 * np.matmul(np.transpose(delta_train),train_images_matrix[row:row+1])
        else:
            continue
#################################################### TRAIN ACCURACY CALCULATION ###################################

    train_actual = train_y.transpose().reshape(60000,).astype(int)
    cfm_train = confusion_matrix(train_actual, train_predicted)
    #print cfm
    diagonal_sum_train =  sum(np.diag(cfm_train))
    #print diagonal_sum_train
    accuracy_train[i] = (diagonal_sum_train/60000.00)*100
    print accuracy_train[i]

#################################################### TEST  CALCULATION ###########################################

    for row_test in range(len(test_images_matrix)):
        y_final_test = np.zeros([1,10],dtype=int)
        y_test = np.matmul(test_images_matrix[row_test:row_test+1],np.transpose(weights))
        maxindex_test = np.argmax(y_test, axis = 1)
        y_final_test[0,maxindex_test[0]] = 1
        #delta_test = test_labels_final[row_test:row_test+1] - y_final_test
        index_y_test = y_final_test.argmax(axis = 1)
        test_y[row_test] = index_y_test
        #print test_y[row_test]

#################################################### TEST ACCURACY CALCULATION ####################################

    test_actual = test_y.transpose().reshape(10000,).astype(int)
    cfm_test = confusion_matrix(test_actual, test_predicted)
    diagonal_sum_test =  sum(np.diag(cfm_test))
    #print diagonal_sum_test
    accuracy_test[i] = (diagonal_sum_test/10000.00)*100
    print accuracy_test[i]

####################################################### GRAPH PLOT ################################################
print "Confusion matrix for test matrix with learning rate 0.001"
print cfm_test
print "Confusion matrix for train matrix with learning rate 0.001"
print cfm_train

image = "learning_rate=0.001.png"
plt.title("learning_rate = 0.001")
plt.plot(accuracy_train)
plt.plot(accuracy_test)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.savefig(image)
plt.show()
