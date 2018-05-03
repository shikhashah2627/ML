import numpy as np
import gzip
from sklearn.metrics import confusion_matrix

Learning_rate = 0.1
Momentum = 0.9
Hidden_layer = 20

class Perceptron:
    def _init_(self):
        global train_images, train_labels, train_labels_final, weights2, deltaO, deltaH. /
        weights3,finding_accuracy_train,finding_accuracy_test,
        f = gzip.open("train-images-idx3-ubyte.gz", 'rb')
        train_images = np.frombuffer(f.read(), np.uint8, offset=16)
        train_images = train_images.reshape(-1, 1, 28, 28)
        train_images = train_images.reshape(train_images.shape[0],784)
        train_images = train_images / np.float32(255)
        train_images = np.append(train_images,np.ones((60000,1)),axis = 1)

        f = gzip.open("train-labels-idx1-ubyte.gz", 'rb')
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        train_labels = train_labels.reshape(60000,1)

        train_labels_final = np.full((60000, 10), 0.1, dtype=float)

        for row in range(len(train_labels)):
            index = train_labels[row].astype(int)
            train_labels_final[row][index[0]] += 0.8

        weights2 = np.random.uniform(-0.05, 0.05, size=(785,Hidden_layer))
        deltaO = np.zeros((Hidden_layer+1,10))
        deltaH = np.zeros((Hidden_layer,785))
        weights3 = np.random.uniform(-0.05, 0.05, size=(Hidden_layer+1,10))

        accuracy = 0

        finding_accuracy_train = np.zeros((60000,1))
        finding_accuracy_test = np.zeros((10000,1))
        forward_phase_input_to_hidden = np.zeros(Hidden_layer+1)
        forward_phase_input_to_hidden[0] = 1


        f = gzip.open("test-images.gz", 'rb')
        test_images = np.frombuffer(f.read(), np.uint8, offset=16)
        test_images = test_images.reshape(-1, 1, 28, 28)
        test_images = test_images.reshape(test_images.shape[0],784)
        test_images = np.append(test_images,np.ones((10000,1)),axis = 1)


        f = gzip.open("test-labels.gz", 'rb')
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        test_labels = test_labels.reshape(10000,1)

        test_labels_final = np.full((10000, 10), 0.1, dtype=float)

        for row in range(len(test_labels)):
            index = test_labels[row].astype(int)
            test_labels_final[row][index[0]] += 0.8

    def sigmoid(self,input):
        return ( 1 + np.exp(-input))

    def matrix_mult_input_to_hidden(self,row):
        return np.dot(train_images[row][:],weights2)

    def matrix_mult_hidden_to_output(self,forward_phase_input_to_hidden):
        return np.dot(forward_phase_input_to_hidden,weights3)

    def output_error(self,row,output_sigmoid):
        return output_sigmoid * (1 - output_sigmoid) * (train_labels_final[row][:] - output_sigmoid)

    def hidden_error(self,forward_phase_input_to_hidden,output_error,):
        x = forward_phase_input_to_hidden[1:] * ( 1 - forward_phase_input_to_hidden[1:] )
        y =  np.dot(weights3[1:,:],np.transpose(output_error))
        hidden_error = x * y
        return hidden_error

    def hidden_to_output_weight_update(self,forward_phase_input_to_hidden,output_error):
        step_1 = Learning_rate * np.outer(forward_phase_input_to_hidden,output_error)
        step_2 = Momentum * deltaO
        newdeltaO = step_1 + step_2
        weights3 += newdeltaO
        deltaO = newdeltaO
        return weights3

    def  input_to_hidden_weight_update:
        step_1 = Learning_rate * np.outer(hidden_error,train_images[row][:])
        step_2 = Momentum * deltaH
        newdeltaH = step_1 + step_2
        weights2 += np.transpose(newdeltaH)
        deltaH = newdeltaH
        return weights2

def main():
    for i in range(50):
        class_object = Perceptron()
        for row in range(len(train_images)):
            
            h2 = np.dot(train_images[row][:],weights2)

            forward_phase_input_to_hidden[1:] = 1 / ( 1 + np.exp(-h2))
            h4 = np.dot(forward_phase_input_to_hidden,weights3)
            #print h4.shape
            output_sigmoid = 1 / ( 1 + np.exp(-h4))
            output_error = output_sigmoid * (1 - output_sigmoid) * (train_labels_final[row][:] - output_sigmoid) #1*10

            x = forward_phase_input_to_hidden[1:] * ( 1 - forward_phase_input_to_hidden[1:] )
            y =  np.dot(weights3[1:,:],np.transpose(output_error))
            hidden_error = x * y
            #print "X.shape" , x.shape , "y.shape" , y.shape , "hiddem_error", hidden_error.shape

            step_1 = Learning_rate * np.outer(forward_phase_input_to_hidden,output_error)
            step_2 = Momentum * deltaO
            newdeltaO = step_1 + step_2

            step_1 = Learning_rate * np.outer(hidden_error,train_images[row][:])
            #print step_1.shape , hidden_error.shape ,train_images[row][:].shape
            step_2 = Momentum * deltaH
            newdeltaH = step_1 + step_2

            weights3 += newdeltaO
            deltaO = newdeltaO
            weights2 += np.transpose(newdeltaH)
            deltaH = newdeltaH

            index = np.argmax(output_sigmoid,axis=0)
            finding_accuracy_train[row] =  index


        for row in range(len(test_images)):

            h2 = np.dot(test_images[row][:],weights2)

            forward_phase_input_to_hidden[1:] = 1 / ( 1 + np.exp(-h2))
            h4 = np.dot(forward_phase_input_to_hidden,weights3)
            #print h4.shape
            output_sigmoid = 1 / ( 1 + np.exp(-h4))

            index = np.argmax(output_sigmoid,axis=0)
            finding_accuracy_test[row] =  index



    cfm_train = confusion_matrix(train_labels,finding_accuracy_train)
    diagonal_sum_train =  sum(np.diag(cfm_train))
    accuracy_train = (diagonal_sum_train/60000.00)*100
    print cfm_train
    print accuracy_train


    cfm_test = confusion_matrix(test_labels,finding_accuracy_test)
    diagonal_sum_test =  sum(np.diag(cfm_test))
    accuracy_test = (diagonal_sum_test/10000.00)*100
    print cfm_test
    print accuracy_test
