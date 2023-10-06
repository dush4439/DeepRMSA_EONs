import tensorflow as tf
from tensorflow.keras import layers, Model

class AC_Net(Model):
    def __init__(self, scope, trainer, x_dim_p, x_dim_v, n_actions, num_layers, layer_size, regu_scalar):
        super(AC_Net, self).__init__()

        self.scope = scope
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.x_dim_p = x_dim_p
        self.n_actions = n_actions
        self.x_dim_v = x_dim_v
        self.regu_scalar = regu_scalar
        self.trainer = trainer

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Policy network
        self.policy_net = tf.keras.Sequential()
        for i in range(num_layers):
            self.policy_net.add(layers.Dense(layer_size, activation='elu'))
        self.policy_net.add(layers.Dense(n_actions, activation='softmax'))

        # Value network
        self.value_net = tf.keras.Sequential()
        for i in range(num_layers):
            self.value_net.add(layers.Dense(layer_size, activation='elu'))
        self.value_net.add(layers.Dense(1))

        # Regularizer
        self.regularizer = tf.keras.regularizers.l2(regu_scalar)

    def call(self, inputs):
        policy_input, value_input = inputs

        policy = self.policy_net(policy_input)
        value = self.value_net(value_input)

        return policy, value

    def train(self, policy_input, value_input, actions, actions_onehot, target_v, advantages):
        with tf.GradientTape() as tape:
            policy, value = self.call((policy_input, value_input))

            responsible_outputs = tf.reduce_sum(policy * actions_onehot, axis=1)

            # Loss functions
            regu_loss_policy = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.scope + '/policy'))
            entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-6))
            loss_policy = -tf.reduce_sum(tf.math.log(responsible_outputs + 1e-6) * advantages) - entropy * 0.01 + regu_loss_policy

            regu_loss_value = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.scope + '/value'))
            loss_value = tf.reduce_sum(tf.math.square(target_v - value))

            total_loss = loss_policy + loss_value

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

