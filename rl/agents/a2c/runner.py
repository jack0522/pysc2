import numpy as np
import tensorflow as tf
#引入 NumPy，用於數值計算，主要處理多維數據。
#引入 TensorFlow，用於實現深度學習模型與訓練。
-------------------------------------------------------------------------------------
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES
#從 PySC2 導入 FunctionCall（用於表示遊戲動作）與 FUNCTIONS（遊戲中可用的動作函數）。
#導入 TYPES 作為動作參數的類型集合。
-------------------------------------------------------------------------------------
from rl.pre_processing import Preprocessor
from rl.pre_processing import is_spatial_action, stack_ndarray_dicts
# 從自定義模組 rl.pre_processing 中導入以下工具：
# Preprocessor：用於對遊戲的觀測數據進行預處理。
# is_spatial_action：用於判斷動作參數是否為空間相關。
# stack_ndarray_dicts：堆疊多個數據字典。

-------------------------------------------------------------------------------------

class A2CRunner():    #定義 A2CRunner 類別，用於執行 A2C 強化學習的核心訓練邏輯
  def __init__(self,
               agent,
               envs,
               summary_writer=None,
               train=True,
               n_steps=8,
               discount=0.99):
# 定義類別初始化方法，接收以下參數：
# agent：A2C 代理模型實例。
# envs：多進程環境實例。
# summary_writer：用於記錄訓練分數。
# train：布林值，是否進行訓練。
# n_steps：每批次的步數。
# discount：折扣因子，用於計算回報。
------------------------------------------------------------                 
    """
    Args:
      agent: A2CAgent instance.
      envs: SubprocVecEnv instance.
      summary_writer: summary writer to log episode scores.
      train: whether to train the agent.
      n_steps: number of agent steps for collecting rollouts.
      discount: future reward discount.
    """
    self.agent = agent
    self.envs = envs
    self.summary_writer = summary_writer
    self.train = train
    self.n_steps = n_steps
    self.discount = discount
#將參數值存儲為類別屬性
------------------------------------------------------------------------
    self.preproc = Preprocessor(self.envs.observation_spec()[0])
#初始化 Preprocessor，並傳入環境的觀測規格，用於處理原始觀測數據。
-------------------------------------------------------------------------
    self.episode_counter = 0
    self.cumulative_score = 0.0
#初始化累計分數與回合計數器。
-------------------------------------------------------------------------
  def reset(self):
    obs_raw = self.envs.reset()
    self.last_obs = self.preproc.preprocess_obs(obs_raw)
# 定義 reset 方法：
#   重置環境並獲取原始觀測數據。
#   使用 Preprocessor 對觀測數據進行預處理，存儲於 self.last_obs。
--------------------------------------------------------------------------
  def get_mean_score(self):
    return self.cumulative_score / self.episode_counter

#計算累計分數的平均值，用於評估代理模型的表現。
---------------------------------------------------------------------------
  def _summarize_episode(self, timestep):     #定義內部方法，用於總結每一回合。
    score = timestep.observation["score_cumulative"][0]
    #從觀測數據中提取總分數。
----------------------------------------------------------------------------
    if self.summary_writer is not None:
      summary = tf.Summary()
      summary.value.add(tag='sc2/episode_score', simple_value=score)
      self.summary_writer.add_summary(summary, self.episode_counter)
#如果 summary_writer 不為空，則將分數記錄到 TensorFlow 日誌中（TF 1.x 方式）。
----------------------------------------------------------------------------
    print("episode %d: score = %f" % (self.episode_counter, score))
    self.episode_counter += 1
    return score
#在終端輸出回合數與分數，並更新回合計數器。

-----------------------------------------------------------------------------
  def run_batch(self, train_summary=False):    #定義 run_batch 方法，用於執行多個步驟，並根據需要進行訓練。
    """Collect trajectories for a single batch and train (if self.train).

    Args:
      train_summary: return a Summary of the training step (losses, etc.).

    Returns:
      result: None (if not self.train) or the return value of agent.train.
    """
    shapes = (self.n_steps, self.envs.n_envs)
    values = np.zeros(shapes, dtype=np.float32)
    rewards = np.zeros(shapes, dtype=np.float32)
    dones = np.zeros(shapes, dtype=np.float32)
    all_obs = []
    all_actions = []
    all_scores = []
# #初始化用於存儲數據的陣列與列表：
#   values：存放價值函數估計。
#   rewards：存放每步的獎勵。
#   dones：存放每步是否完成。
#   all_obs/all_actions：存放所有觀測數據與動作。
#   all_scores：存放所有分數。
-----------------------------------------------------------------
    last_obs = self.last_obs
#獲取之前的觀測數據。
-----------------------------------------------------------------
    for n in range(self.n_steps):
      actions, value_estimate = self.agent.step(last_obs)
      actions = mask_unused_argument_samples(actions)
# 循環執行 n_steps：
#   獲取代理模型的動作與價值估計。
#   遮蔽未使用的參數。
------------------------------------------------------------------
      size = last_obs['screen'].shape[1:3]

      values[n, :] = value_estimate
      all_obs.append(last_obs)
      all_actions.append(actions)
#存儲價值估計與觀測數據。
------------------------------------------------------------------
      pysc2_actions = actions_to_pysc2(actions, size)
      obs_raw = self.envs.step(pysc2_actions)
      last_obs = self.preproc.preprocess_obs(obs_raw)

#將動作轉換為 PySC2 格式，執行環境步驟，並預處理新觀測數據。
------------------------------------------------------------------
      rewards[n, :] = [t.reward for t in obs_raw]
      dones[n, :] = [t.last() for t in obs_raw]
#提取回報與完成標記。
  ----------------------------------------------------------------
      for t in obs_raw:
        if t.last():
          score = self._summarize_episode(t)
          self.cumulative_score += score
  #如果回合完成，總結分數並更新累計分數。
------------------------------------------------------------------

    self.last_obs = last_obs

    next_values = self.agent.get_value(last_obs)

    returns, advs = compute_returns_advantages(
        rewards, dones, values, next_values, self.discount)

    actions = stack_and_flatten_actions(all_actions)
    obs = flatten_first_dims_dict(stack_ndarray_dicts(all_obs))
    returns = flatten_first_dims(returns)
    advs = flatten_first_dims(advs)

    if self.train:
      return self.agent.train(
          obs, actions, returns, advs,
          summary=train_summary)

    return None


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


def flatten_first_dims(x):
  new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
  return x.reshape(*new_shape)


def flatten_first_dims_dict(x):
  return {k: flatten_first_dims(v) for k, v in x.items()}


def stack_and_flatten_actions(lst, axis=0):
  fn_id_list, arg_dict_list = zip(*lst)
  fn_id = np.stack(fn_id_list, axis=axis)
  fn_id = flatten_first_dims(fn_id)
  arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
  arg_ids = flatten_first_dims_dict(arg_ids)
  return (fn_id, arg_ids)
