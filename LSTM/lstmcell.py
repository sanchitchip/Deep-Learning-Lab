import numpy as np
from tensorflow.contrib.rnn import LayerRNNCell

class LSTM(LayerRNNCell):
    def __init__(self , num_units , reuse=None,name=None):
        super (LayerRNNCell , self). __init__ ( reuse=reuse , name=name, dtype=dtype)
        self._num_units = num_units
    @property
    def state_size(self):
        return ( self._num_units , self._num_units)
    @property
    def output_size(self):
        return self._num_units

    def build(self,input_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                       str(inputs_shape))
            #raise ValueError(”Expected inputs.shape[−1] to be known, saw shape: %s” % inputs_shape)
        input_size = inputs_shape[1].value
        # input
        self.W_i = tf.Variable(tf.truncated_normal([input_size, self._num_units]))
        self.U_i = tf.Variable(tf.truncated_normal([self._num_units, self._num_units]))
        self.b_i = tf.Variable(tf.zeros([self._num_units]))
        # forget
        self.W_f = tf.Variable(tf.truncated_normal([input_size, self._num_units]))
        self.U_f = tf.Variable(tf.truncated_normal([self._num_units, self._num_units]))
        self.b_f = tf.Variable(tf.zeros([self._num_units]))
        # output
        self.W_o = tf.Variable(tf.truncated_normal([input_size, self._num_units]))
        self.U_o = tf.Variable(tf.truncated_normal([self._num_units, self._num_units]))
        self.b_o = tf.Variable(tf.zeros([self._num_units]))
        # cell
        self.W_c = tf.Variable(tf.truncated_normal([input_size, self._num_units]))
        self.U_c = tf.Variable(tf.truncated_normal([self._num_units, self._num_units]))
        self.b_c = tf.Variable(tf.zeros([self._num_units]))

        self.built=True

    def call(self,inputs,state):
        c_prev,h_prev = state
        # input gate
        i_g = tf.sigmoid(tf.matmul(x,self.W_i)+tf.matmul(h_prev,self.U_i)+ self.b_i)
        # forget gate
        f_g = tf.sigmoid(tf.matmul(x,self.W_f)+tf.matmul(h_prev,self.U_f)+ self.b_f)
        # output gate
        o_g = tf.sigmoid(tf.matmul(x,self.W_o)+tf.matmul(h_prev,self.U_o)+ self.b_o)
        c_t = tf.nn.tanh(tf.matmul(x,self.W_c)+tf.matmul(h_prev,self.U_c)+ self.b_c)
        new_c =  f_g*c_prev + i_g*c_t
        new_h =o_g*tf.nn.tanh(new_c)

        new_state = (new_c,new_h)
        
        return new_h,new_state
