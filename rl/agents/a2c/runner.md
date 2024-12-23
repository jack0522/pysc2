# A2CRunner 說明與逐行解析

此文件詳細說明 `runner.py` 的功能與實現，並逐行解釋每段程式碼的用途和邏輯。

---

## 導入模組
```python
import numpy as np
import tensorflow as tf
```
### **解釋:**

#### 1.`import numpy as np`:
- 引入 NumPy，用於數值運算和多維陣列操作。
#### 2.`import tensorflow as tf`:
- 引入 TensorFlow，用於構建和訓練深度學習模型。

```python
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES
```
### **解釋:**

#### 1.`FunctionCall`:
- 表示 StarCraft II 中的動作結構。
#### 2.`FUNCTIONS`:
- 包含 StarCraft II 中所有可用的動作函數。
#### 3.`TYPES`作為`ACTION_TYPES`:
- 包含動作參數的類型集合，取別名為 `ACTION_TYPES`。
  
```python
from rl.pre_processing import Preprocessor
from rl.pre_processing import is_spatial_action, stack_ndarray_dicts
```
### **解釋:**

#### 1.`Preprocessor`:
- 用於處理從環境獲得的原始觀測數據。
#### 2.`is_spatial_action`:
- 判斷動作是否涉及空間參數。
#### 3.`stack_ndarray_dicts`:
- 堆疊多個數據字典，便於批量處理。
## 類別定義
```python
class A2CRunner():
```
### **解釋:**

- 定義類別 `A2CRunner`，負責 A2C 演算法的執行與訓練。
## 初始化方法
```python
def __init__(self, agent, envs, summary_writer=None, train=True, n_steps=8, discount=0.99):
```
### **解釋:**

#### 1.功能:
-初始化類別，設置執行環境和核心參數。
#### 2.參數:
- `agent`：A2C 強化學習代理。
- `envs`：多環境實例。
- `summary_writer`：TensorFlow 摘要記錄器。
- `train`：布林值，是否進行訓練。
- `n_steps`：每批次的步數。
- `discount`：折扣因子，用於計算未來回報。
```python
self.agent = agent
self.envs = envs
self.summary_writer = summary_writer
self.train = train
self.n_steps = n_steps
self.discount = discount
self.preproc = Preprocessor(self.envs.observation_spec()[0])
self.episode_counter = 0
self.cumulative_score = 0.0
```
### **解釋:**

#### 1.`self.agent`:
- 保存代理模型。
#### 2.`self.envs`:
- 保存多環境實例。
#### 3.`self.summary_writer`:
- 保存 TensorFlow 摘要記錄器。
#### 4.`self.preproc`:
- 初始化觀測數據的預處理器。
#### 5.`self.episode_counter`:
- 記錄完成的回合數。
#### 6.`self.cumulative_score`:
- 累計所有回合的分數。
## `reset` 方法
```python
def reset(self):
    obs_raw = self.envs.reset()
    self.last_obs = self.preproc.preprocess_obs(obs_raw)
```
### **解釋:**

#### 1.功能:
-重置環境並初始化觀測數據。
#### 2.步驟:
- `self.envs.reset()`：重置所有環境，獲取初始觀測數據。
- `self.preproc.preprocess_obs(obs_raw)`：對觀測數據進行預處理。
## `_summarize_episode` 方法
```python
def _summarize_episode(self, timestep):
    score = timestep.observation["score_cumulative"][0]
    if self.summary_writer is not None:
        summary = tf.Summary()
        summary.value.add(tag='sc2/episode_score', simple_value=score)
        self.summary_writer.add_summary(summary, self.episode_counter)

    print("episode %d: score = %f" % (self.episode_counter, score))
    self.episode_counter += 1
    return score
```
### **解釋:**

#### 1.功能:
-記錄回合分數並輸出到終端。
#### 2.邏輯:
- 提取累積分數。
- 如果配置了 summary_writer，將分數記錄到 TensorFlow 摘要。
- 更新回合計數器並返回分數。
## `run_batch` 方法
```python
def run_batch(self, train_summary=False):
    shapes = (self.n_steps, self.envs.n_envs)
    values = np.zeros(shapes, dtype=np.float32)
    rewards = np.zeros(shapes, dtype=np.float32)
    dones = np.zeros(shapes, dtype=np.float32)
    all_obs = []
    all_actions = []
    all_scores = []
```
### **解釋:**

#### 1.功能:
-執行多步環境交互並收集數據。
#### 2.初始化:
-定義存儲數據的容器，包括價值估計、獎勵、完成標記等。
## 主循環:
```python
for n in range(self.n_steps):
    actions, value_estimate = self.agent.step(last_obs)
    actions = mask_unused_argument_samples(actions)
    size = last_obs['screen'].shape[1:3]

    values[n, :] = value_estimate
    all_obs.append(last_obs)
    all_actions.append(actions)

    pysc2_actions = actions_to_pysc2(actions, size)
    obs_raw = self.envs.step(pysc2_actions)
    last_obs = self.preproc.preprocess_obs(obs_raw)
    rewards[n, :] = [t.reward for t in obs_raw]
    dones[n, :] = [t.last() for t in obs_raw]

    for t in obs_raw:
        if t.last():
            score = self._summarize_episode(t)
            self.cumulative_score += score
```
### **解釋:**

#### 1.功能:
-循環執行 n_steps 步的環境交互，並收集數據。
#### 2.邏輯:
-代理根據觀測生成動作和價值估計。
-將動作轉換為 PySC2 格式，執行環境步驟，並記錄觀測數據、回報和完成標記。
-如果回合結束，記錄分數並更新累積分數。
# 工具函數完整解析

---

## `compute_returns_advantages`

### 程式碼：
```python
def compute_returns_advantages(rewards, dones, values, next_values, discount):
    """Compute returns and advantages from received rewards and value estimates.

    Args:
        rewards: array of shape [n_steps, n_env] containing received rewards.
        dones: array of shape [n_steps, n_env] indicating whether an episode is
          finished after a time step.
        values: array of shape [n_steps, n_env] containing estimated values.
        next_values: array of shape [n_env] containing estimated values after the
          last step for each environment.
        discount: scalar discount for future rewards.

    Returns:
        returns: array of shape [n_steps, n_env]
        advs: array of shape [n_steps, n_env]
    """
    returns = np.zeros([rewards.shape[0] + 1, rewards.shape[1]])

    returns[-1, :] = next_values
    for t in reversed(range(rewards.shape[0])):
        future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
        returns[t, :] = rewards[t, :] + future_rewards

    returns = returns[:-1, :]
    advs = returns - values

    return returns, advs
```
### **解釋:**

#### 1.功能:
-計算每一步的折扣回報（returns）和優勢（advantages）。
#### 2.邏輯:
-初始化 `returns`，大小為 `[n_steps + 1, n_env]`，多出的行用於輔助計算。
-設置最後一步的回報為 `next_values`。
-從後向前迭代計算回報：
```python
future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
returns[t, :] = rewards[t, :] + future_rewards
```
-如果某步回合結束（dones[t, :] == 1），後續回報不計入。
#### 3.輸出:
-返回回報和優勢，用於更新模型。
## `actions_to_pysc2`

### 程式碼：
```python
def actions_to_pysc2(actions, size):
    """Convert agent action representation to FunctionCall representation."""
    height, width = size
    fn_id, arg_ids = actions
    actions_list = []
    for n in range(fn_id.shape[0]):
        a_0 = fn_id[n]
        a_l = []
        for arg_type in FUNCTIONS._func_list[a_0].args:
            arg_id = arg_ids[arg_type][n]
            if is_spatial_action[arg_type]:
                arg = [arg_id % width, arg_id // height]
            else:
                arg = [arg_id]
            a_l.append(arg)
        action = FunctionCall(a_0, a_l)
        actions_list.append(action)
    return actions_list
```
### **解釋:**

#### 1.功能:
-將代理模型生成的動作轉換為 PySC2 可接受的 `FunctionCall` 格式。
#### 2.邏輯:
- 提取動作函數 `ID（fn_id）`和參數 `ID（arg_ids）`。
- 對於每個動作函數：
    -  遍歷其參數類型（`arg_type`）。
    -  若參數為空間類型，轉換為二維座標：
    ```python
    arg = [arg_id % width, arg_id // height]
    ```
    -  否則直接保留參數 ID。
    -  組裝為 FunctionCall，並加入動作列表。
#### 3.輸出:
-返回 actions_list，包含所有動作的 FunctionCall 表示。
## `mask_unused_argument_samples`

### 程式碼：
```python
def mask_unused_argument_samples(actions):
    """Replace sampled argument id by -1 for all arguments not used
    in a steps action (in-place).
    """
    fn_id, arg_ids = actions
    for n in range(fn_id.shape[0]):
        a_0 = fn_id[n]
        unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[a_0].args)
        for arg_type in unused_types:
            arg_ids[arg_type][n] = -1
    return (fn_id, arg_ids)
```
### **解釋:**

#### 1.功能:
-將動作中未使用的參數遮蔽為 -1，以優化計算。
#### 2.邏輯:
-遍歷每個動作，確定未使用的參數類型：
```python
unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[a_0].args)
```
-將這些類型的參數值設為 -1。
#### 3.輸出:
-返回更新後的動作。
## `flatten_first_dims`

### 程式碼：
```python
def flatten_first_dims(x):
    new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
    return x.reshape(*new_shape)
```
### **解釋:**

#### 1.功能:
-將輸入陣列的前兩個維度展平為一維。
#### 2.邏輯:
-計算展平後的新形狀：
```python
new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
```
-使用 reshape 方法重塑數據。
## `flatten_first_dims_dict`

### 程式碼：
```python
def flatten_first_dims_dict(x):
    return {k: flatten_first_dims(v) for k, v in x.items()}
```
### **解釋:**

#### 1.功能:
-將字典中所有數組的前兩個維度展平。
#### 2.邏輯:
-遍歷字典的每個鍵值，對數組應用 flatten_first_dims。
## `stack_and_flatten_actions`

### 程式碼：
```python
def stack_and_flatten_actions(lst, axis=0):
    fn_id_list, arg_dict_list = zip(*lst)
    fn_id = np.stack(fn_id_list, axis=axis)
    fn_id = flatten_first_dims(fn_id)
    arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
    arg_ids = flatten_first_dims_dict(arg_ids)
    return (fn_id, arg_ids)
```
### **解釋:**

#### 1.功能:
-堆疊多個動作數據，並展平維度
#### 2.邏輯:
-將動作列表拆分為函數 ID 和參數字典列表。
-堆疊函數 ID 並展平成一維。
-堆疊參數字典並展平所有數據。








