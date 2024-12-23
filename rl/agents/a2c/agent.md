# A2Cagent 說明與逐行解析

此文件詳細說明 `agent.py` 的功能與實現，並逐行解釋每段程式碼的用途和邏輯。

---

## 導入模組
```python
import os

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.distributions import Categorical

from pysc2.lib.actions import TYPES as ACTION_TYPES

from rl.networks.fully_conv import FullyConv
from rl.util import safe_log, safe_div
```
### **解釋:**

#### 1.`import os`:
- 用於進行文件與目錄的操作，如模型保存和加載。
#### 2.`tensorflow`:
- 用於構建和訓練神經網絡。
- `layers` 提供方便的神經網絡層構建工具。
- `Categorical`: 定義離散分佈，用於動作採樣。
#### 3.`pysc2.lib.actions`:
-導入動作參數的類型集合。
#### 4.`rl.networks.fully_conv`:
-導入 `FullyConv` 類別，構建卷積網絡。
#### 5.`rl.util`:
導入工具函數 `safe_log` 和 `safe_div`，分別用於安全取對數和除法。
## A2CAgent 類別
```python
class A2CAgent():
    """A2C agent.

    Run build(...) first, then init() or load(...).
```
### **解釋:**

#### 1.`class A2CAgent`:
- 定義 A2C 強化學習代理。
- 用於執行網絡構建（`build`）、初始化（`init`）或從檔案加載模型（ `load`）。
## 初始化方法
```python
def __init__(self,
             sess,
             network_cls=FullyConv,
             network_data_format='NCHW',
             value_loss_weight=0.5,
             entropy_weight=1e-3,
             learning_rate=7e-4,
             max_gradient_norm=1.0,
             max_to_keep=5):
    self.sess = sess
    self.network_cls = network_cls
    self.network_data_format = network_data_format
    self.value_loss_weight = value_loss_weight
    self.entropy_weight = entropy_weight
    self.learning_rate = learning_rate
    self.max_gradient_norm = max_gradient_norm
    self.train_step = 0
    self.max_to_keep = max_to_keep
```
### **解釋:**

#### 1.功能:
- 初始化 A2C 代理，設置網絡、優化器和訓練相關參數。
#### 2.參數:
- `sess`：TensorFlow 會話，用於執行計算圖。
- `network_cls`：網絡類型，默認為 `FullyConv`。
- `network_data_format`：數據格式，支持 `NCHW` 和 `NHWC`。
- `value_loss_weight`：價值損失的權重。
- `entropy_weight`：策略熵的權重，用於鼓勵探索。。
- `learning_rate`：學習率。
- `max_gradient_norm`：梯度裁剪的最大範圍。
- `max_to_keep`：保存的最大檢查點數量。






















