import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow                    as     tf
import tensorflow.keras              as     tk
import tensorflow_probability        as     tfp
from   tensorflow.keras              import Model
from   tensorflow.keras.layers       import Dense
from   tensorflow.keras.initializers import Orthogonal, LecunNormal
from   tensorflow.keras.optimizers   import Adam

# Define alias
tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tfd = tfp.distributions

###############################################
### Neural network for mu and sg prediction
class nn(Model):
    def __init__(self, arch, i_dim, o_dim, act, last, lr):
        super(nn, self).__init__()

        # Initialize network as empty list
        self.lr   = lr
        self.arch = arch
        self.net  = []

        # Define hidden layers
        for layer in range(len(arch)):
            self.net.append(Dense(arch[layer],
                                  kernel_initializer=LecunNormal(),
                                  activation=act))

        # Define last layer
        self.net.append(Dense(o_dim,
                              kernel_initializer=LecunNormal(),
                              activation=last))

        # Initialize with dummy forward pass
        self.call(tf.zeros([1,i_dim]))

        # Define optimizer
        self.opt = Adam(learning_rate = self.lr)
        zero_grads = [tf.zeros_like(w) for w in self.trainable_weights]
        self.opt.apply_gradients(zip(zero_grads, self.trainable_weights))

        # Save network initial weights
        self.net_weights = self.get_weights()

        # Save optimizer initial config
        self.opt_weights = self.opt.get_weights()
        self.opt_config  = self.opt.get_config()

    # Network forward pass
    @tf.function
    def call(self, var):

        # Copy input
        x = var

        # Compute output
        for layer in range(len(self.net)):
            x = self.net[layer](x)

        return tf.cast(x, tf.float32)

    # Reset weights
    def reset(self):

        self.opt.set_weights(self.opt_weights)
        self.opt.from_config(self.opt_config)
        self.set_weights(self.net_weights)
