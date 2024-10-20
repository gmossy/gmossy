from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
    # def __init__(self, state_size, action_size, num_units = 300):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            num_units (int): Number of units per layer
        """
        self.state_size = state_size
        self.action_size = action_size
        # Initialize any other variables here
        # self.num_units = num_units   
        # neural network layer parameters:
        # dropout = 0.3
        
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=40,kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Dropout(0.5)(net_states)
        net_states = layers.Activation("relu")(net_states)
        
        net_states = layers.Dense(units=30, kernel_regularizer=layers.regularizers.l2(1e-6))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Dropout(0.5)(net_states)
        net_states = layers.Activation("relu")(net_states)
        
        net_states = layers.Dense(units=30, kernel_regularizer=layers.regularizers.l2(1e-6))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Dropout(0.5)(net_states)
        net_states = layers.Activation("relu")(net_states)
        
        
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=30,kernel_regularizer=layers.regularizers.l2(1e-6))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.Dropout(0.5)(net_actions)
        net_actions = layers.Activation("relu")(net_actions)
       
        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        
        # Combine state and action pathway
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values',kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)
        #  Q_values = layers.Dense(units=1, name='q_values')(net)
        
        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001) # learning rate, normal setting is actor lr=0.0001, critic lr=0.001
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)