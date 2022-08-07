import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# input data
wvlt_amp = np.load('wvlt_amp.npy')

Vp_t = np.load('Vp_t.npy')
Vs_t = np.load('Vs_t.npy')
Den_t = np.load('Den_t.npy')

Vp0_t = np.load('Vp0_t.npy')
Vs0_t = np.load('Vs0_t.npy')
Den0_t = np.load('Den0_t.npy')

#%% setting the parameters
dz = 5
dt = 0.004
nt = len(Vp_t)


#depth to time conversion
T = dt * np.linspace(0.0, nt-1, num = nt)

#seismic scale
scale_seismic = 100

Vp_t = np.reshape(Vp_t,(-1,1))
Vs_t = np.reshape(Vs_t,(-1,1))
Den_t = np.reshape(Den_t,(-1,1))

Vp0_t = np.reshape(Vp0_t,(-1,1))
Vs0_t = np.reshape(Vs0_t,(-1,1))
Den0_t = np.reshape(Den0_t,(-1,1))

# acoustic impedance
AI_t = Vp_t * Den_t * 10 ** (-7)
AI0_t = Vp0_t * Den0_t * 10 ** (-7)

Phi0_t = (AI0_t - 0.3) / 5
Phi_t = (AI_t - 0.3) / 5

#%%
# create the wavelet map
wvlt_map = np.zeros((len(wvlt_amp),len(wvlt_amp)))

half = int(nt/2)

for i in range(half):
    wvlt_map[0:len(wvlt_amp)+i-(half)+1,i] = wvlt_amp[(half)-i-1:]
    
for i in range(half,len(wvlt_amp)):
    wvlt_map[i-len(wvlt_amp)+half:,i] = wvlt_amp[0:len(wvlt_amp)-i+half-1]


ref_t = (AI_t[1:] -  AI_t[:-1]) / (AI_t[:-1] +  AI_t[1:])

PP_t = np.matmul(wvlt_map, ref_t)

PP_t = np.reshape(PP_t,[-1,1])

PP_plus = np.zeros((1,1))

PP_t = np.vstack((PP_plus, PP_t))

#%%
# add noise to seismic
Noise = 1
SNR = 30

if Noise == 1:
    for i in range(PP_t.shape[1]):
        SignalRMS = np.sqrt(np.mean(np.power(PP_t[:, i], 2)))
        NoiseTD   = np.random.randn(len(PP_t[:, i]),1)
        NoiseRMS  = np.sqrt(np.mean(np.power(NoiseTD, 2)));
        New = np.reshape(PP_t[:, i], (-1, 1)) + (SignalRMS/NoiseRMS) * np.power(10, -SNR/20) * NoiseTD
        PP_t[:, i] = New[:, 0]
#%%
       
X_train_init = np.reshape(PP_t,[-1,1])
Y_train = np.reshape(PP_t[1:,],[-1,1])

window_width = 10

from expand_dims import expand_dims

X_train = expand_dims(X_train_init, window_width)   

#%% Define the cnn model

AI0_t_tf = tf.constant(AI0_t, dtype=tf.float32, name='AI0_t_tf')
AI_t_tf  = tf.constant(AI_t, dtype=tf.float32, name='AI_t_tf')

Phi0_t_tf = tf.constant(Phi0_t, dtype=tf.float32, name='Phi0_t_tf')
Phi_t_tf = tf.constant(Phi_t, dtype=tf.float32, name='Phi_t_tf')

wvlt_map_tf = tf.constant(wvlt_map, dtype=tf.float32, name='wvlt_map_tf')
# input x and y
input_data   = tf.placeholder(tf.float32, shape = X_train.shape)
output_data  = tf.placeholder(tf.float32, shape = [None, None])

input_dim = 1
num_filters = 12
dropout_prob = 0.5

## neural network layers
conv1       = tf.layers.conv1d(input_data, num_filters, 1, strides=1, padding = 'valid', kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer='zeros')
conv1_norm  = tf.layers.batch_normalization(conv1)
conv1_activ = tf.nn.sigmoid(conv1_norm)

conv2       = tf.layers.conv1d(conv1_activ, num_filters, 3, padding = 'valid', kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer='zeros')
conv2_norm  = tf.layers.batch_normalization(conv2)
#conv2_norm  = conv2
conv2_activ = tf.nn.sigmoid(conv2_norm)

dropout     = tf.nn.dropout(conv2_activ, keep_prob = 1 - dropout_prob/2)

flat_layer  = tf.layers.flatten(dropout)
dense       = tf.layers.dense(flat_layer, units = 4 * num_filters, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer='zeros')
dense_norm  = tf.layers.batch_normalization(dense)
#dense_norm  = dense
dense_activ = tf.nn.sigmoid(dense_norm)
#dense_dp    = tf.nn.dropout(dense_activ, keep_prob = 1 - dropout_prob)

y_pred      = tf.layers.dense(dense_activ, units = 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer='zeros')
y_pred      = tf.nn.tanh(y_pred)

Phi_t_predict = Phi0_t_tf + y_pred

AI_t_predict = 5 * Phi_t_predict + 0.3

ref_t_predict = (AI_t_predict[1:] -  AI_t_predict[:-1]) / (AI_t_predict[:-1] +  AI_t_predict[1:])

syn_PP_t = tf.matmul(wvlt_map_tf, ref_t_predict)


data_misfit = 100 * tf.reduce_mean(tf.square(syn_PP_t - output_data))
model_misfit = 1 * tf.reduce_mean(tf.square(Phi_t_predict - Phi0_t_tf))
up = tf.reduce_mean(tf.square(Phi_t_predict[0] - Phi0_t_tf[0]))
bottom = tf.reduce_mean(tf.square(Phi_t_predict[-1] - Phi0_t_tf[-1]))
bound = 1 * (up + bottom) / 2

loss =  data_misfit + model_misfit

optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

#%%
plt.close('all')

iter = 5001

l = np.zeros((iter,1))
data_error = np.zeros((iter,1))
model_error = np.zeros((iter,1))
AI_predict_iter = np.zeros((len(AI0_t),iter))
Phi_predict_iter = np.zeros((len(AI0_t),iter))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
      
    for step in range(iter):
        dic = {input_data: X_train, output_data: Y_train}
        
        _, AI_predict, l[step], model_error[step], Phi_predict, contrast = sess.run([train, AI_t_predict, loss, model_misfit, Phi_t_predict, y_pred], feed_dict = dic)
        
        
        ref_predict = (AI_predict[1:] -  AI_predict[:-1]) / (AI_predict[:-1] +  AI_predict[1:])
        
        top = AI_predict[0] - AI0_t[0]
        bottom = AI_predict[-1] - AI0_t[-1]
        
        AI_predict = AI_predict - (top + bottom) / 2
        
        AI_predict_iter[:,step] = AI_predict.ravel()
        
        top = Phi_predict[0] - Phi0_t[0]
        bottom = Phi_predict[-1] - Phi0_t[-1]
        
        Phi_predict = Phi_predict - (top + bottom) / 2
        
        Phi_predict_iter[:,step] = Phi_predict.ravel()
        
        if step % 1000 == 0:
            print('step {}'.format(step))
            print(l[step])
#%%
for i in range(3000,iter):
    fig = plt.figure(num = 8, figsize=(4, 8))
    fig.set_facecolor('white')
    plt.plot(AI_predict_iter[:,i].ravel(), T, '-b', label = 'Inversion')
    plt.plot(AI_t.ravel(), T, '-r', label = 'Truth')
    plt.plot(AI0_t.ravel(), T, '--k', label = 'Prior')
    plt.ylim(T[0],T[-1])
    plt.gca().invert_yaxis()
    plt.title('AI (Iter = %i)' %i)
    plt.grid(True)
    plt.ylabel('TWT (sec)')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
    legend_x = 1
    legend_y = 1
    plt.legend(loc='upper left', bbox_to_anchor=(legend_x, legend_y))
    plt.pause(0.000001)
    plt.clf()
    
    fig = plt.figure(num = 9, figsize=(4, 8))
    fig.set_facecolor('white')
    plt.plot(Phi_predict_iter[:,i].ravel(), T, '-b', label = 'Inversion')
    plt.plot(Phi_t.ravel(), T, '-r', label = 'Truth')
    plt.plot(Phi0_t.ravel(), T, '--k', label = 'Prior')
    plt.ylim(T[0],T[-1])
    plt.gca().invert_yaxis()
    plt.title('Porosity (Iter = %i)' %i)
    plt.grid(True)
    plt.ylabel('TWT (sec)')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
    legend_x = 1
    legend_y = 1
    plt.legend(loc='upper left', bbox_to_anchor=(legend_x, legend_y))
    plt.pause(0.000001)
    plt.clf()
