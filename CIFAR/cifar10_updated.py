#@title
import tensorflow as tf
import numpy as np
import pdb
from matplotlib import pyplot as plt
from google.colab import files

## IMPORTANT NOTE:
## This is the code for last question and so therefore has got tr_dat,tr_lab as the training set(both train and validation) while test set for computation of accuracy.
## for computation of validation accuracy please replace tr_dat,tr_lab with X_train,Y_train in the batch for loop and X_test,Y_test with X_val,Y_val.
## Also change vprob=1 for checking of results of 1 answer as that would be equavalent to hacing no dropout.
#Regards


def Normalize(input):
    return input/255
    
def one_hot(input):
    ## creating ine hot label matrix with zeros
    vlab = np.zeros((input.shape[0],10))
    for idx,lab in enumerate(input):
        vlab[idx][lab] = 1
    return vlab

def cnnmodel(input,vProb):
    W_conv1 = tf.Variable(tf.truncated_normal([3,3,3,32],stddev=.09))
    bias1 = tf.Variable(tf.zeros(shape=(32,)))
    conv1 = tf.nn.conv2d(input,W_conv1,strides=[1,1,1,1],padding="SAME")
    conv1 = conv1+bias1
    relu1 = tf.nn.relu(conv1)
    #2nd convnet
    W_conv2 = tf.Variable(tf.truncated_normal([3,3,32,32],stddev=.09))
    bias2 = tf.Variable(tf.zeros(shape=(32,)))
    conv2 = tf.nn.conv2d(relu1,W_conv2,strides=[1,1,1,1],padding="SAME")
    conv2 = conv2+bias2
    relu2 = tf.nn.relu(conv2)
    #max pool layer after 2 CNNs
    pool1 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    dp1 = tf.nn.dropout(pool1, vProb)
   
    # 3rd CNN
    W_conv3 = tf.Variable(tf.truncated_normal([3,3,32,64],stddev=.09))
    bias3 = tf.Variable(tf.zeros(shape=(64,)))
    conv3 = tf.nn.conv2d(dp1,W_conv3,strides=[1,1,1,1],padding="SAME")
    conv3 = conv3+bias3
    relu3 = tf.nn.relu(conv3)
    # 4th cnn
    W_conv4 = tf.Variable(tf.truncated_normal([3,3,64,64],stddev=.09))
    bias4 = tf.Variable(tf.zeros(shape=(64,)))
    conv4 = tf.nn.conv2d(relu3,W_conv4,strides=[1,1,1,1],padding="SAME")
    conv4 = conv4+bias4
    relu4 = tf.nn.relu(conv4)
    # max pool layer after 4th convnet
    pool2 = tf.nn.max_pool(relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    dp2 = tf.nn.dropout(pool2, vProb)
       # flattening the featuee maps.
#    vflat = tf.contrib.layers.flatten(dp2)
# 2 max pool layer makes it 8*8*64.
    vflat = tf.reshape(dp2,[-1,4096])
    # Fully Connected NN layer with output 512

    #     fnn1 = tf.contrib.layers.fully_connected(vflat,num_outputs=512)
    W_fc1 = tf.Variable(tf.truncated_normal([4096,512],stddev=.09))
    b_fc1 = tf.Variable(tf.zeros(shape = (512,)))
    fnn1 = tf.matmul(vflat,W_fc1) + b_fc1
    relu5 = tf.nn.relu(fnn1)
    dp3 = tf.nn.dropout(relu5, vProb)
       # no need to do softmax as it automatically gets done during the loss function
    # add last layer with (num_classes) output
    W_fc2 = tf.Variable(tf.truncated_normal([512,10],stddev=.09))
    b_fc2 = tf.Variable(tf.zeros(shape = (10,)))
    final_out = tf.matmul(dp3,W_fc2) + b_fc2

#    final_out = tf.contrib.layers.fully_connected(dp3,num_outputs=10)
    return final_out

# to test model  on cpu or if learning is happening.
## to-do add bias[migth be affecting acc.]::
def cnnmodel_small(input):
    W_conv1 = tf.Variable(tf.truncated_normal([3,3,3,32],stddev=.01))
#    b_conv = tf.Variable(tf.zeros(shape=(32,3)))
    A_conv = tf.nn.conv2d(input,W_conv1,strides=[1,1,1,1],padding="VALID")
    A_conv = tf.nn.relu(A_conv)
    A_conv = tf.nn.max_pool(A_conv,ksize=[1,2,2,1],strides=[1,1,1,1],padding="VALID")
    ## to flatten
    vflat = tf.contrib.layers.flatten(A_conv)
    final_out = tf.contrib.layers.fully_connected(vflat,num_outputs=10)
    
    return final_out

    

    
tf.reset_default_graph()
tf.set_random_seed(22)
vData = tf.keras.datasets.cifar10.load_data()

#normalize
#vData[0][0] = 
vShape_tr= vData[0][0].shape[0]
# generating indices shuffled randomly
#vInd = np.random.choice(vShape_tr,vShape_tr,replace=False)
## shuffled train set
sh_idx=np.arange(vShape_tr)
np.random.shuffle(sh_idx)
tr_dat = Normalize(vData[0][0])
tr_lab = one_hot(vData[0][1])
X_train = tr_dat[sh_idx[:49000]]
Y_train = tr_lab[sh_idx[:49000]]
#X_train = Normalize(X_train)
#Y_train = one_hot(Y_train)
## shuffled validation set
X_val = tr_dat[sh_idx[49000:]]
#X_val = Normalize(X_val)
Y_val = tr_lab[sh_idx[49000:]]
#Y_val = one_hot(Y_val)

# test set
X_test = Normalize(vData[1][0])
Y_test = one_hot(vData[1][1])

print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)
#pdb.set_trace()


## Hyperparam initialize
lr = .001
epoch=250
batch_size=32
dp_prob=.75
# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x_tr')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y_tr')
vprob = tf.placeholder(tf.float32)
## loss
print("big model")
prediction = cnnmodel(x,vprob)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=prediction))

## optimizer
#optimizer = tf.train.AdamOptimizer(lr)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
vtrain = optimizer.minimize(loss)

## accuracy computation
vcount = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(vcount,tf.float32))


##
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_ep = []
    for j in range(epoch):
        # arranging indices randomly and iterationg on batches using that.
        # vind = np.random.choice(X_train.shape[0],X_train.shape[0],replace=False)
        print("Epoch number is {}".format(j))

        sh_idx2 = np.arange(tr_dat.shape[0])
        np.random.shuffle(sh_idx2)
        batch_train_loss = []
        for i in range(0,tr_dat.shape[0],32):
            vstart = i
            vend = i+32
            # incase if the last batch is <32 change its end point to remove the error.
            if i+32>tr_dat.shape[0]:
                vend = tr_dat.shape[0]
                
            vInput = tr_dat[sh_idx2[vstart:vend]]
            vLabel = tr_lab[sh_idx2[vstart:vend]]
            _, vloss = sess.run([vtrain,loss],                                   
                                   feed_dict={x:vInput,y:vLabel,vprob:dp_prob})
            batch_train_loss.append(vloss)
            
        train_loss = np.mean(batch_train_loss)
                    
        # computing val in batches too no shuffling needed here.
        batch_test_acc =[]
        batch_test_loss=[]
        for i in range(0,X_test.shape[0],32):
            vstart = i
            vend=i+32
            if i+32>X_test.shape[0]:
                vend = X_test.shape[0]
                
            vInput = X_test[vstart:vend]
            vLabel = Y_test[vstart:vend]
            vloss,vacc = sess.run([loss,accuracy],                                   
                                   feed_dict={x:vInput,y:vLabel,vprob:1})
            batch_test_acc.append(vacc)
            batch_test_loss.append(vloss)

        vAcc = np.mean(batch_test_acc)
        test_loss = np.mean(batch_test_loss)  
        test_ep.append(vAcc*100)    
        print("train loss is {}".format(train_loss))
        print("test loss is {}".format(test_loss))
        print("test accuracy is {}".format(vAcc))
    # percentage conversion.
    plt.plot(np.arange(epoch),test_ep)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("test accuracy wrt Epoch")
    plt.show()
