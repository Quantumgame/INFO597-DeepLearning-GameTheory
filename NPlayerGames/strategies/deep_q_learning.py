import random
import math
import tensorflow as tf
import numpy as np
from collections import deque


class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, given_layers=None):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"

        # print('MLP init scope', self.scope)

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        with tf.variable_scope(self.scope):
            if given_layers is not None:
                self.input_layer = given_layers[0]
                self.layers      = given_layers[1:]
            else:
                # print('var scope', self.scope)
                self.input_layer = Layer(input_sizes, hiddens[0], scope=self.scope+"_input_layer")
                self.layers = []

                for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
                    self.layers.append(Layer(h_from, h_to, scope="hidden_layer_%d" % (l_idx,)))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        with tf.variable_scope(self.scope):
            hidden = self.input_nonlinearity(self.input_layer(xs))
            for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
                hidden = nonlinearity(layer(hidden))
            return hidden

    def variables(self):
        res = self.input_layer.variables()
        for layer in self.layers:
            res.extend(layer.variables())
        return res

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        scope = self.scope + "_copy"
        nonlinearities = [self.input_nonlinearity] + self.layer_nonlinearities
        given_layers = [self.input_layer.copy()] + [layer.copy() for layer in self.layers]
        return MLP(self.input_sizes, self.hiddens, nonlinearities, scope=scope,
                given_layers=given_layers)

class TargetMLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, given_layers=None):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"

        # print('MLP init scope', self.scope)

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        with tf.variable_scope(self.scope):
            if given_layers is not None:
                self.input_layer = given_layers[0]
                self.layers      = given_layers[1:]
            else:
                # print('var scope', self.scope)
                self.input_layer = Layer(input_sizes, hiddens[0], scope=self.scope+"_input_layer_copy")
                self.layers = []

                for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
                    self.layers.append(Layer(h_from, h_to, scope="hidden_layer_%d_copy" % (l_idx,)))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        with tf.variable_scope(self.scope):
            hidden = self.input_nonlinearity(self.input_layer(xs))
            for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
                hidden = nonlinearity(layer(hidden))
            return hidden

    def variables(self):
        res = self.input_layer.variables()
        for layer in self.layers:
            res.extend(layer.variables())
        return res

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        scope = self.scope + "_copy"
        nonlinearities = [self.input_nonlinearity] + self.layer_nonlinearities
        given_layers = [self.input_layer.copy()] + [layer.copy() for layer in self.layers]
        return MLP(self.input_sizes, self.hiddens, nonlinearities, scope=scope,
                given_layers=given_layers)


class DiscreteDeepQ(object):
    def __init__(self, observation_size,
                       num_actions,
                       observation_to_actions,
                       target_actions,
                       optimizer,
                       session,
                       exploration_random_prob=1.0,
                       exploitation_random_prob=0.05,
                       exploration_period=1000,
                       store_every_nth=5,
                       train_every_nth=5,
                       minibatch_size=32,
                       discount_rate=0.95,
                       max_experience=30000,
                       target_network_update_rate=0.01,
                       summary_writer=None):
        """Warning, this class is a modification of:
        https://github.com/nivwusquorum/tensorflow-deepq/blob/master/tf_rl/controller/discrete_deepq.py

        Initialized the Deepq object.
        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        Parameters
        -------
        observation_size : int
            length of the vector passed as observation
        num_actions : int
            number of actions that the model can execute
        observation_to_actions: dali model
            model that implements activate function
            that can take in observation vector or a batch
            and returns scores (of unbounded values) for each
            action for each observation.
            input shape:  [batch_size, observation_size]
            output shape: [batch_size, num_actions]
        optimizer: tf.solver.*
            optimizer for prediction error
        session: tf.Session
            session on which to execute the computation
        random_action_probability: float (0 to 1)
        exploration_period: int
            probability of choosing a random
            action (epsilon form paper) annealed linearly
            from 1 to random_action_probability over
            exploration_period
        store_every_nth: int
            to further decorrelate samples do not all
            transitions, but rather every nth transition.
            For example if store_every_nth is 5, then
            only 20% of all the transitions is stored.
        train_every_nth: int
            normally training_step is invoked every
            time action is executed. Depending on the
            setup that might be too often. When this
            variable is set set to n, then only every
            n-th time training_step is called will
            the training procedure actually be executed.
        minibatch_size: int
            number of state,action,reward,newstate
            tuples considered during experience reply
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        max_experience: int
            maximum size of the reply buffer
        target_network_update_rate: float
            how much to update target network after each
            iteration. Let's call target_network_update_rate
            alpha, target network T, and network N. Every
            time N gets updated we execute:
                T = (1-alpha)*T + alpha*N
        summary_writer: tf.train.SummaryWriter
            writer to log metrics
        """
        # memorize arguments
        self.observation_size          = observation_size
        self.num_actions               = num_actions

        self.q_network                 = observation_to_actions
        self.target_q_network          = target_actions
        self.optimizer                 = optimizer
        self.s                         = session

        self.exploration_random_prob   = exploration_random_prob
        self.exploitation_random_prob  = exploitation_random_prob
        self.exploration_period        = exploration_period
        self.store_every_nth           = store_every_nth
        self.train_every_nth           = train_every_nth
        self.minibatch_size            = minibatch_size
        self.discount_rate             = tf.constant(discount_rate)
        self.max_experience            = max_experience
        self.target_network_update_rate = \
                tf.constant(target_network_update_rate)

        # deepq state
        self.actions_executed_so_far = 0
        self.experience = deque()

        self.iteration = 0
        self.summary_writer = summary_writer

        self.number_of_times_store_called = 0
        self.number_of_times_train_called = 0

        self.create_variables()

    def linear_annealing(self, n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / (total)

    def create_variables(self):
        #self.target_q_network    = self.q_network.copy() # scope="target_network"

        # FOR REGULAR ACTION SCORE COMPUTATION
        with tf.name_scope("taking_action"):
            self.observation        = tf.placeholder(tf.float32, (None, self.observation_size), name="observation")
            self.action_scores      = tf.identity(self.q_network(self.observation), name="action_scores")
            tf.summary.histogram("action_scores", self.action_scores)
            self.predicted_actions  = tf.argmax(self.action_scores, dimension=1, name="predicted_actions")

        with tf.name_scope("estimating_future_rewards"):
            # FOR PREDICTING TARGET FUTURE REWARDS
            self.next_observation          = tf.placeholder(tf.float32, (None, self.observation_size), name="next_observation")
            self.next_observation_mask     = tf.placeholder(tf.float32, (None,), name="next_observation_mask")
            self.next_action_scores        = tf.stop_gradient(self.target_q_network(self.next_observation))
            tf.summary.histogram("target_action_scores", self.next_action_scores)
            self.rewards                   = tf.placeholder(tf.float32, (None,), name="rewards")
            target_values                  = tf.reduce_max(self.next_action_scores, reduction_indices=[1,]) * self.next_observation_mask
            self.future_rewards            = self.rewards + self.discount_rate * target_values

        with tf.name_scope("q_value_precition"):
            # FOR PREDICTION ERROR
            self.action_mask                = tf.placeholder(tf.float32, (None, self.num_actions), name="action_mask")
            self.masked_action_scores       = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1,])
            temp_diff                       = self.masked_action_scores - self.future_rewards
            self.prediction_error           = tf.reduce_mean(tf.square(temp_diff))
            gradients                       = self.optimizer.compute_gradients(self.prediction_error)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 5), var)
            # Add histograms for gradients.
            for grad, var in gradients:
                tf.summary.histogram(var.name, var)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradients', grad)
            self.train_op                   = self.optimizer.apply_gradients(gradients)

        # UPDATE TARGET NETWORK
        with tf.name_scope("target_network_update"):
            self.target_network_update = []
            for v_source, v_target in zip(self.q_network.variables(), self.target_q_network.variables()):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_network_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update)

        # summaries
        tf.summary.scalar("prediction_error", self.prediction_error)

        self.summarize = tf.summary.merge_all()
        self.no_op1    = tf.no_op()

    def action(self, observation):
        """Given observation returns the action that should be chosen using
        DeepQ learning strategy. Does not backprop."""
        assert len(observation.shape) == 1, \
                "Action is performed based on single observation."

        self.actions_executed_so_far += 1
        exploration_p = self.linear_annealing(self.actions_executed_so_far,
                                              self.exploration_period,
                                              self.exploration_random_prob,
                                              self.exploitation_random_prob)

        if random.random() < exploration_p:
            return random.randint(0, self.num_actions - 1)
        else:
            # print('self.predicted_actions', self.predicted_actions)
            # print('obs', observation[np.newaxis,:])
            return self.s.run(self.predicted_actions, {self.observation: observation[np.newaxis,:]})[0]

    def store(self, observation, action, reward, newobservation):
        """Store experience, where starting with observation and
        execution action, we arrived at the newobservation and got thetarget_network_update
        reward reward
        If newstate is None, the state/action pair is assumed to be terminal
        """
        if self.number_of_times_store_called % self.store_every_nth == 0:
            self.experience.append((observation, action, reward, newobservation))
            if len(self.experience) > self.max_experience:
                self.experience.popleft()
        self.number_of_times_store_called += 1

    def training_step(self):
        """Pick a self.minibatch_size exeperiences from reply buffer
        and backpropage the value function.
        """
        if self.number_of_times_train_called % self.train_every_nth == 0:
            if len(self.experience) <  self.minibatch_size:
                return

            # sample experience.
            samples   = random.sample(range(len(self.experience)), self.minibatch_size)
            samples   = [self.experience[i] for i in samples]

            # bach states
            states         = np.empty((len(samples), self.observation_size))
            newstates      = np.empty((len(samples), self.observation_size))
            action_mask    = np.zeros((len(samples), self.num_actions))

            newstates_mask = np.empty((len(samples),))
            rewards        = np.empty((len(samples),))

            for i, (state, action, reward, newstate) in enumerate(samples):
                states[i] = state
                action_mask[i] = 0
                #print(action)
                action_mask[i][action] = 1
                rewards[i] = reward
                if newstate is not None:
                    newstates[i] = newstate
                    newstates_mask[i] = 1
                else:
                    newstates[i] = 0
                    newstates_mask[i] = 0


            calculate_summaries = self.iteration % 100 == 0 and \
                    self.summary_writer is not None

            cost, _, summary_str = self.s.run([
                self.prediction_error,
                self.train_op,
                self.summarize if calculate_summaries else self.no_op1,
            ], {
                self.observation:            states,
                self.next_observation:       newstates,
                self.next_observation_mask:  newstates_mask,
                self.action_mask:            action_mask,
                self.rewards:                rewards,
            })

            self.s.run(self.target_network_update)

            if calculate_summaries:
                self.summary_writer.add_summary(summary_str, self.iteration)

            self.iteration += 1

        self.number_of_times_train_called += 1


def linear_annealing(n, total=8000, p_initial=1.0, p_final=0.05):
    """Linear annealing between p_initial and p_final
    over total steps - computes value at step n"""
    if n >= total:
        return p_final
    else:
        return p_initial - (n * (p_initial - p_final)) / (total)

def base_name(var):
    """Extracts value passed to name= when creating a variable"""
    return var.name.split('/')[-1].split(':')[0]


class Layer(object):
    def __init__(self, input_sizes, output_size, scope):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.scope       = scope or "Layer"

        # print('scope: ', self.scope)
        with tf.variable_scope(self.scope):
            self.Ws = []
            for input_idx, input_size in enumerate(input_sizes):
                W_name = "W_%d" % (input_idx,)
                # print(W_name)
                W_initializer =  tf.random_uniform_initializer(
                        -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
                W_var = tf.get_variable(W_name, (input_size, output_size), initializer=W_initializer)
                self.Ws.append(W_var)
            self.b = tf.get_variable("b", (output_size,), initializer=tf.constant_initializer(0))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))
        with tf.variable_scope(self.scope):
            return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

    def variables(self):
        return [self.b] + self.Ws

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return Layer(self.input_sizes, self.output_size, scope=sc)
