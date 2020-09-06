# wrapper for python environment
from tf_agents.environments import tf_py_environment

# dqn modules
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

# training, metrics and reports
import matplotlib.pyplot as plt
import tensorflow as tf

# environment
from CardGameEnv import CardGameEnv

# quiet warnings about features being deprecated (for now)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DQN_trainer:
    def __init__(self, *args, **kwargs):
        self.env = tf_py_environment.TFPyEnvironment(CardGameEnv())
        self.q_net = q_network.QNetwork(self.env.observation_spec(), self.env.action_spec())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.train_step_counter = tf.Variable(0)
        self.agent = dqn_agent.DqnAgent(self.env.time_step_spec(),
                           self.env.action_spec(),
                           q_network=self.q_net,
                           optimizer=self.optimizer,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=self.train_step_counter)

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self.agent.collect_data_spec,
                                                                            batch_size=self.env.batch_size,
                                                                            max_length=100000)
        self.num_iterations = 10000
    
    def compute_avg_return(self, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = self.env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = self.env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    def collect_step(self):
        time_step = self.env.current_time_step()
        action_step = self.agent.policy.action(time_step)
        next_time_step = self.env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)
    
    def train(self):
        # initialize the agent
        self.agent.initialize()

        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(5)
        returns = [avg_return]

        collect_steps_per_iteration = 1
        batch_size = 64
        dataset = self.replay_buffer.as_dataset(num_parallel_calls=3, 
                                            sample_batch_size=batch_size, 
                                            num_steps=2).prefetch(3)
        iterator = iter(dataset)
        self.env.reset()

        for _ in range(batch_size):
            self.collect_step()

        for _ in range(self.num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(collect_steps_per_iteration):
                self.collect_step()

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            # Print loss every 200 steps.
            if step % 200 == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            # Evaluate agent's performance every 1000 steps.
            if step % 1000 == 0:
                avg_return = self.compute_avg_return(5)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
        
        self.plot_results(returns)

    def plot_results(self, returns):
        plt.figure(figsize=(12,8))
        iterations = range(0, self.num_iterations + 1, 1000)
        plt.plot(iterations, returns)
        plt.title('DQN Agent Results')
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.show()

testRun = DQN_trainer()

testRun.train()