import tensorflow as tf
from tensorflow.keras import layers, Model

'''
Final code for Actor Critic network in python.
Use the Tensorflow 2.x
'''

class AC_Net(Model):
    def __init__(self, scope, trainer, x_dim_p, x_dim_v, n_actions, num_layers, layer_size, regu_scalar):
        super(AC_Net, self).__init__()
        """Initializes the Actor-Critic network with specified parameters.

        Args:
            scope: A unique identifier for this network.
            trainer: Optimizer for training.
            x_dim_p: Dimensionality of the policy input.
            x_dim_v: Dimensionality of the value input.
            n_actions: Number of possible actions.
            num_layers: Number of hidden layers in the neural networks.
            layer_size: Number of neurons in each hidden layer.
            regu_scalar: Regularization strength.
        """
        # Initialize the Actor-Critic network with specified parameters
        self.scope = scope # A unique identifier for this network
        self.num_layers = num_layers # Number of hidden layers in the neural networks
        self.layer_size = layer_size # Number of neurons in each hidden layer
        self.x_dim_p = x_dim_p # Dimensionality of the policy input
        self.n_actions = n_actions # Number of possible actions
        self.x_dim_v = x_dim_v # Dimensionality of the value input
        self.regu_scalar = regu_scalar # Regularization strength
        self.trainer = trainer # Optimizer for training

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Define the Policy Network (Actor)
        self.policy_net = tf.keras.Sequential()
        for i in range(num_layers):
            self.policy_net.add(layers.Dense(layer_size, activation='elu'))
        self.policy_net.add(layers.Dense(n_actions, activation='softmax'))

        # Define the Value Network (Critic)
        self.value_net = tf.keras.Sequential()
        for i in range(num_layers):
            self.value_net.add(layers.Dense(layer_size, activation='elu'))
        self.value_net.add(layers.Dense(1))

        # Define L2 regularization for both policy and value networks
        self.regularizer = tf.keras.regularizers.l2(regu_scalar)

    def call(self, inputs):
        """
        Computes policy and value for the given inputs.

        Args:
            inputs: A tuple of (policy_input, value_input).

        Returns:
            A tuple of (policy, value)."""
        
        policy_input, value_input = inputs

        policy = self.policy_net(policy_input)
        value = self.value_net(value_input)

        return policy, value

    def train(self, policy_input, value_input, actions, actions_onehot, target_v, advantages):
        """
        Trains the policy and value networks.

        Args:
            policy_input: A tensor of shape (batch_size, state_dim) representing the policy inputs.
            value_input: A tensor of shape (batch_size, state_dim) representing the value inputs.
            actions: A tensor of shape (batch_size, action_dim) representing the actions taken.
            actions_onehot: A tensor of shape (batch_size, action_dim) representing the one-hot encoded actions.
            target_v: A tensor of shape (batch_size,) representing the target values.
            advantages: A tensor of shape (batch_size,) representing the advantages.

        Returns:
            None."""

        # Define L2 regularization for both policy and value networks
        with tf.GradientTape() as tape:
            # Forward pass to compute policy and value predictions
            policy, value = self.call((policy_input, value_input))

            # Calculate responsible outputs (policy for selected actions)
            responsible_outputs = tf.reduce_sum(policy * actions_onehot, axis=1)

            # Calculate regularization losses for policy and value networks
            regu_loss_policy = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.scope + '/policy'))
            entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-6))
            loss_policy = -tf.reduce_sum(tf.math.log(responsible_outputs + 1e-6) * advantages) - entropy * 0.01 + regu_loss_policy

            regu_loss_value = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.scope + '/value'))
            loss_value = tf.reduce_sum(tf.math.square(target_v - value))


            # Calculate the total loss as a combination of policy and value losses
            total_loss = loss_policy + loss_value

        # Compute gradients of the total loss with respect to network parameters
        grads = tape.gradient(total_loss, self.trainable_variables)

        # Apply the computed gradients using the optimizer
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        