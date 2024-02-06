import gym
import tensorflow as tf
import numpy as np
from time import sleep

# NN defs
n_inputs = 4
n_hidden_layers = 10
n_outputs = 2
initializer = tf.keras.initializers.LecunNormal(seed=42)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(n_inputs)),
    tf.keras.layers.Dense(n_inputs, activation='relu', kernel_initializer=initializer),           
    tf.keras.layers.Dense(n_hidden_layers, activation='relu', kernel_initializer=initializer), 
    tf.keras.layers.Dense(n_hidden_layers, activation='relu', kernel_initializer=initializer), 
    tf.keras.layers.Dense(n_outputs, activation='relu', kernel_initializer=initializer)
    ])

model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
        )

config = model.get_config()
print(config["layers"][0]["config"])

env = gym.make('Reacher-v4', render_mode='rgb_array')

# Train on random data
trials = 10
max_steps = 200
training_x, training_y = [], []
scores = []
for trial in range(trials):
    score = 0
    observation, info = env.reset(seed=42)
    _x, _y = [], []
    for step in range(max_steps):
        action = env.action_space.sample()
        _x.append(observation[0:4])
        _y.append(action)
        observation, reward, done, _, _ = env.step(action)
        score += reward
        if done: break
    print('score: ', score)
    scores.append(score)
    training_x += _x
    training_y += _y
training_x, training_y = np.array(training_x), np.array(training_y)

print('Average: {}'.format(np.mean(scores)))
print('Median: {}'.format(np.median(scores)))
print('training data shapes: ', training_x.shape, training_y.shape)
model.fit(training_x, training_y, epochs=5)
env.close()

def policy(observation):
    x = observation[0:4].reshape(1, 4)
    y = model.predict(x)
    return y

# Gym session
n_epi = 100
max_steps = 500
avg_steps = []
discount_factor = 0.1

# Run the learning loop
env = gym.make('Reacher-v4', render_mode='human')
for epi in range(n_epi): #episodes
    print('episode: ', epi)
    observation, info = env.reset(seed=42)
    
    # Save data for training
    states = []
    actions = []
    rewards = []

    for step in range(max_steps):
        print('step: ', step)
        env.render()

        states.append(observation[0:4])
        action = policy(observation)[0]  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
    
        if terminated or truncated:
            avg_steps.append(step)
            break

    # parse data
    states, actions, rewards = np.array(states), np.array(actions), np.array(rewards)
    print('states, actions, rewards shape', states.shape, actions.shape, rewards.shape)

    # compute expected returns
    discounted_rewards = np.zeros(rewards.shape)
    acc = 0
    for i, r in enumerate(rewards[::-1]):
        acc = r + discount_factor*acc
        discounted_rewards[i] = acc
    
    #standardize
    returns = (discounted_rewards - discounted_rewards.mean())/discounted_rewards.std()

    # Convert the lists of states, actions, and discounted rewards to tensors
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards)
    
    # Train the policy network using the REINFORCE algorithm
    with tf.GradientTape() as tape:
        # Get the action probabilities from the policy network
        action_probs = model(states)
        # Calculate the loss
        loss = tf.cast(tf.math.log(tf.gather(action_probs,actions,axis=1,batch_dims=1)),tf.float64)* discounted_rewards
        loss = -tf.reduce_sum(loss)

        # Calculate the gradients and update the policy network
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


env.close()
