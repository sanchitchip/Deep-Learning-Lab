import numpy as np
import pickle
import pdb
import tensorflow as tf
from google.colab import drive
from matplotlib import pyplot as plt
drive.mount('/content/drive')

def generate_batches(text,batch_size,sequence_length):
    block_length = len(text)//batch_size
    batches = []

    for i in range(0, block_length,sequence_length):
        batch=[]
        for j in range(batch_size):
            start = j*block_length + i
            end = min(start+sequence_length,j*block_length+block_length)
            batch.append(text[start:end])
        batches.append(np.array(batch,dtype=int))
        
    return batches


## to reduce the noise and make it little less random choosing from top 5 predictions:
def top_n(preds,char_size,top_n=5):
    vP = np.squeeze(preds)
    # making other than top_n pred 0
    vP[np.argsort(vP)[:-top_n]] = 0
    # recomputing prob so as to put in np random choice function.
    vP = vP/np.sum(vP)
    # prediction of next char
    next_char = np.random.choice(char_size,1,p=vP)[0]
    return next_char

def generate_char(model_saved,n_class,n_samples=256, lstm_size=256, vInit="the "):
    vsamp = [i for i in vInit]
    model = CharLSTM(n_class, lstm_size=lstm_size, to_sample=True)
    vpreds = tf.nn.softmax(model.prediction)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_saved)
        new_state = sess.run(model.initial_state)

        for i in vInit:
            x = np.zeros((1, 1))
            x[0,0] = char_int[i]
        #    print(x)
            feed = {model.X_input: x,
                    model.initial_state: new_state,
                    model.dropout_prob:1}
#            preds, new_state = sess.run([model.prediction, model.final_state], 
#                                         feed_dict=feed)
            preds, new_state = sess.run([vpreds, model.final_state], 
                                         feed_dict=feed)

        c = top_n(preds, n_class)
        vsamp.append(int_char[c])
        
        for i in range(n_samples):
            x[0,0] = c
            feed = {model.X_input: x,
                    model.initial_state: new_state,
                    model.dropout_prob:1}
#            preds, new_state = sess.run([model.prediction, model.final_state], 
#                                         feed_dict=feed)

            preds, new_state = sess.run([vpreds, model.final_state], 
                                         feed_dict=feed)
            c = top_n(preds, n_class)
            vsamp.append(int_char[c])
        
    return ''.join(vsamp)

tf.reset_default_graph()

#vFile = "/Users/sanchit/Sanchit/Study/DLL/Assignment/3/monte.txt"
vFile = "/content/drive/My Drive/Colab Notebooks/monte.txt"
with open(vFile,encoding='utf8') as f:
    #    vocab = pickle.load(f)
    raw_text = f.read().lower()

# finding unique charecters in text file and converting into list for further computations.
vlist = set(raw_text)
# sorting of characters before doing one hot encoding


# remove characters with frequence less than 1000.
char_freq = {}
for chr in vlist:
    char_freq[chr] = 0
for chr in raw_text:
    char_freq[chr]+=1
# remove char with low fequency
updated_text = []
for chr in raw_text:
    if char_freq[chr]>1000:
        updated_text.append(chr)

## converting to string just to check if it's looking right.
updated_text = "".join(updated_text)

vlist = set(updated_text)
vchar = sorted(vlist)
len_char = len(vchar)
print("Number of Unique Characters are {}".format(len_char))

# choosing interger to represent each character in a dictionary
char_int = {c:i for i,c in enumerate(vchar)}
int_char = {i:c for i,c in enumerate(vchar)}

vencoded = [char_int[vch] for vch in updated_text]
vtarget = vencoded[1:]
# hyperparameters.
batch_size = 32
vseq_len=256
lstm_size =256
learning_rate = .01

class CharLSTM:
    
    def __init__(self, n_class, batch_size=16, n_subseq=256,lstm_size=256, n_layers=2, learning_rate=0.01, to_sample=False,grad_clp=5):
      tf.reset_default_graph()  

        # For eval phase.
      if to_sample == True:
          batch_size, n_subseq = 1, 1
      else:
          batch_size, n_subseq = batch_size, n_subseq

      ## inputs and there one_hot vector
      self.X_input = tf.placeholder(tf.int64,shape=(batch_size,n_subseq))
      # one-hot encoding of X_input
      ### can be source for error watch out:::::: checked works perfect.
      X = tf.one_hot(self.X_input,n_class)

      self.Y_target = tf.placeholder(tf.int64,shape=(batch_size,n_subseq))
      # one-hot encoding of X_input
      ### can be source for error watch out::::checked:fine
      Y = tf.one_hot(self.Y_target,n_class)
      ## dropour probabibility placeholder
      self.dropout_prob = tf.placeholder(tf.float32)

      ## basic LSTM cell.
      def basic_cell(dropout_prob):
          vlstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=256)
          drop = tf.contrib.rnn.DropoutWrapper(vlstm, output_keep_prob=dropout_prob)
          return drop


      cell = tf.nn.rnn_cell.MultiRNNCell([basic_cell(self.dropout_prob) for _ in range(n_layers)],state_is_tuple=True)
      self.initial_state= cell.zero_state(batch_size,tf.float32) 

      lstm_output, next_iter_state = tf.nn.dynamic_rnn(cell=cell,                                                     
                                                inputs=X,
                                                initial_state=self.initial_state)
      self.final_state = next_iter_state
      lin_output = tf.concat(lstm_output, axis=1)
      x = tf.reshape(lin_output, [-1, lstm_size])
      with tf.variable_scope('softmax'):
        s_w = tf.Variable(tf.truncated_normal([lstm_size,n_class],stddev=.1))
        s_b = tf.Variable(tf.zeros([len_char]))
      # changed from working lin_output to x
      self.prediction = tf.matmul(x,s_w)+s_b
      # loss function.
      y_shaped = tf.reshape(Y,self.prediction.get_shape())
      ## loss part
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y_shaped, self.prediction))
      
      # optimization using gradient clipping to control gradient explode.
      # fetching trainable parameters
      tr_vars = tf.trainable_variables()
      #
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tr_vars), grad_clp)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      self.train = optimizer.apply_gradients(zip(grads, tr_vars))
     # optimizer = tf.train.AdamOptimizer(learning_rate)
 #     self.train = optimizer.minimize(self.loss)




#tf.reset_default_graph()
vbatch_input = generate_batches(vencoded,batch_size,vseq_len)
vbatch_input = np.array(vbatch_input[:-1])
vbatch_target= generate_batches(vtarget,batch_size,vseq_len)
vbatch_target = np.array(vbatch_target[:-1])

# model init
model = CharLSTM(len_char, batch_size=batch_size, n_subseq=vseq_len,
                lstm_size=lstm_size,learning_rate=learning_rate)
epochs=10
keep_prob=.5
saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
    loss_plot = []
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = []
        for i in range(len(vbatch_input)):
            ## only taking till 2nd last value
            feed_dict = {model.X_input:vbatch_input[i],
                         model.Y_target:vbatch_target[i],
                         model.initial_state:new_state,
                         model.dropout_prob:keep_prob}
            batch_loss,new_state,_ = sess.run([model.loss,model.final_state,model.train],feed_dict)
            loss.append(batch_loss)
            if(i%100 ==0):
                print("Epoch: {}/{}.....".format(e,epochs))
                print("Batch_loss:  {}.....".format(batch_loss))  
        loss = np.mean(loss)                  
        loss_plot.append(loss)
    saver.save(sess, "model.ckpt")    
#print("here")
m_save = "model.ckpt"

for i in range(5):
  vgenerated = generate_char(m_save,n_class = len_char)
  print(vgenerated)


plt.plot(np.arange(epochs),loss_plot)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train loss wrt Epoch")
plt.show()


## done added gradient clipping,dropout,
## need to add temperature::no time do 1.9
