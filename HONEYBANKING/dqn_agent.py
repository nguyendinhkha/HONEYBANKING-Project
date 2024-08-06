import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        self.n_entries += 1 

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]

# [3] Memory: Lưu trữ các trải nghiệm bằng cách sử dụng SumTree để cho phép phát lại trải nghiệm ưu tiên.
class Memory:
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0

    def __init__(self, capacity, state_size):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.state_size = state_size

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, experience)
    
    # [4] Sample: Lấy ngẫu nhiên một lô các trải nghiệm từ bộ nhớ để huấn luyện mô hình.
    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total()
        ISWeights = np.power(self.tree.n_entries * sampling_probabilities, -1)
        ISWeights /= ISWeights.max()

        # Ensure the data is structured correctly
        for i in range(len(batch)):
            if len(batch[i]) != 5:  # state, action, reward, next_state, done
                print(f"Unexpected data format at index {i}: {batch[i]}")
                print(f"Expected length: 5")
                raise ValueError("Invalid data format in the memory.")

        return idxs, batch, ISWeights

    # [7] Cập nhật lô: Các ưu tiên của các trải nghiệm trong bộ nhớ phát lại được cập nhật dựa trên kết quả huấn luyện.
    def batch_update(self, tree_idx, abs_errors):
        abs_errors = np.asarray(abs_errors).flatten()  # Ensure abs_errors is a flattened array
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(int(ti), float(p))

    def _get_priority(self, error):
        return (error + 0.01) ** 0.6


class DQNAgent:
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(5000, state_size)
        self.gamma = 0.94    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.predictions = []
        self.outcomes = []

        # Use consistent naming for attack types
        self.attack_types = ['sql_injection', 'xss', 'csrf']
        for attack in self.attack_types:
            setattr(self, f'y_true_{attack}', [])
            setattr(self, f'y_pred_{attack}', [])

    def reset_metrics(self):
        for attack in self.attack_types:
            setattr(self, f'y_true_{attack}', [])
            setattr(self, f'y_pred_{attack}', [])

    def update_metrics(self, attack_type, true_label, predicted_label):
        attack = attack_type.lower().replace(' ', '_')
        if attack in self.attack_types:
            getattr(self, f'y_true_{attack}').append(true_label)
            getattr(self, f'y_pred_{attack}').append(predicted_label)
        print(f"Updated metrics for {attack}: y_true - {getattr(self, f'y_true_{attack}')}, y_pred - {getattr(self, f'y_pred_{attack}')}")

    def calculate_metrics(self, y_true, y_pred):
        if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
            print(f"y_true: {y_true}, y_pred: {y_pred}")
            print("Empty or mismatched y_true and y_pred lists. Cannot calculate metrics.")
            logging.warning("Empty or mismatched y_true and y_pred lists. Cannot calculate metrics.")
            return float('nan'), float('nan'), float('nan'), float('nan')

        # Convert y_true and y_pred to binary values
        y_true = [1 if value > 0.5 else 0 for value in y_true]
        y_pred = [1 if value > 0.5 else 0 for value in y_pred]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        logging.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        return accuracy, precision, recall, f1


    def calculate_all_metrics(self):
        metrics = {}
        for attack in self.attack_types:
            y_true = getattr(self, f'y_true_{attack}')
            y_pred = getattr(self, f'y_pred_{attack}')
            metrics[attack] = self.calculate_metrics(y_true, y_pred)
        return metrics

    def plot_metrics(self):
        metrics = self.calculate_all_metrics()
        for attack_type, (accuracy, precision, recall, f1) in metrics.items():
            if not np.isnan(accuracy):
                plt.figure(figsize=(8, 6))
                metric_values = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

                plt.bar(metric_values.keys(), metric_values.values(), color=['blue', 'orange', 'green', 'red'])
                plt.xlabel('Metrics')
                plt.ylabel('Values')
                plt.title(f'DQN Agent Performance Metrics for {attack_type}')
                plt.ylim(0, 1)
                plt.savefig(f'/home/kali/NT505.O21.ATCL_HONEYBANKING_NguyenDinhKha_20520562/HONEYBANKING/metrics_plot_{attack_type}.png')  # Save the plot as a file
                plt.close()  # Close the plot to avoid display issues
            else:
                logging.warning(f"Các chỉ số cho {attack_type} có giá trị NaN, biểu đồ không được tạo.")
        return "Các chỉ số đã được trả về"


    @tf.function(reduce_retracing=True)
    def predict_model(self, states):
        return self.model(states, training=False)
    
    @tf.function(reduce_retracing=True)
    def train_model(self, states, target, sample_weight):
        self.model.fit(states, target, sample_weight=sample_weight, epochs=1, verbose=0)

    def _build_model(self):
        model = Sequential()
        model.add(tf.keras.Input(shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model
    # [8] Mạng mục tiêu được cập nhật định kỳ để khớp với mạng chính sách nhằm ổn định quá trình huấn luyện.
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # [2] Hành động được chọn sẽ được thực hiện trong môi trường, dẫn đến một phần thưởng và trạng thái tiếp theo.
    # Remember: Agent lưu trữ trải nghiệm (trạng thái, hành động, phần thưởng, trạng thái tiếp theo, hoàn thành) vào bộ nhớ.
    def remember(self, state, action, reward, next_state, done, attack_type=None):
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")  # Debugging statement
        self.memory.store((state, action, reward, next_state, done))
        
        if attack_type:
            true_value = 0.04 if reward > 0.0 else 0.0  # Define true_value based on logic
            predicted_value = action
            self.update_metrics(attack_type, true_value, predicted_value)  # Ensure metrics are updated correctly
            print(f"Updated y_true_{attack_type.lower().replace(' ', '_')}: {getattr(self, 'y_true_' + attack_type.lower().replace(' ', '_'))}")  # Debugging statement
            print(f"Updated y_pred_{attack_type.lower().replace(' ', '_')}: {getattr(self, 'y_pred_' + attack_type.lower().replace(' ', '_'))}")  # Debugging statement

    # [1] DQN Agent quyết định hành động nào sẽ thực hiện dựa trên trạng thái hiện tại và mạng chính sách bằng cách sử dụng chiến lược epsilon-greedy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.predict_model(state).numpy()  # Convert to numpy array
        return np.argmax(act_values[0])
    
    # [6]. Huấn luyện mô hình: Mạng nơ-ron được huấn luyện trên lô các trải nghiệm đã được lấy mẫu.
    def replay(self, batch_size):
        if self.memory.tree.n_entries < batch_size:
            return

        tree_idx, minibatch, ISWeights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.vstack(states).astype(np.float32)
        next_states = np.vstack(next_states).astype(np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones).astype(int)
        ISWeights = np.vstack(ISWeights)
        
        # [5]. Các giá trị Q mục tiêu được tính toán bằng cách sử dụng mạng mục tiêu.
        target = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1) * (1 - dones)
        target_f = self.model.predict_on_batch(states)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]

        self.model.train_on_batch(states, target_f, sample_weight=ISWeights)
        self.memory.batch_update(tree_idx, np.abs(target - target_f[np.arange(len(actions)), actions]))


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def memory_size(self):
        return sum(1 for item in self.memory.tree.data if item is not None and np.any(item != 0))
