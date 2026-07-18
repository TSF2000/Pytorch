# 第 12 课 RNN：维度与参数完全吃透

> 配合课程：刘二大人《PyTorch深度学习实践》P12  
> 配合总结：`第12课-循环神经网络基础-总结.md`（本文件更细，专治「懵」）  
> 目标：搞清每个参数、每一维含义；能手写 Cell 版和 RNN 版；能讲清 `hello→ohlol`  
> 建议：边看边在 venv 里跑文中「动手」代码，**每一步都 print shape**

---

## 0. 先建立一张总地图

学 RNN 时脑子里只记四样东西：

| 名字 | 一句话 |
|------|--------|
| **序列** | 一串按顺序来的数据：字符、每天天气…… |
| **时间步 t** | 序列里的第 t 个位置（第几个字符、第几天） |
| **隐藏状态 h** | 网络的「短期记忆」，从上一步传到下一步 |
| **Cell** | 算一步的小模块：吃进 \((x_t, h_{t-1})\)，吐出 \(h_t\) |

整门课就是在重复这句话：

```
同一个 Cell，沿着时间 for 循环用；
每一步用「当前输入 + 上一步记忆」算出「新记忆」。
```

---

## 1. 为什么需要 RNN？（从会的东西推）

### 1.1 你已经会的：一次算完的网络

第 5～11 集大多是：

```
一批样本 → 模型 → 一批输出
```

一张图、一行特征，**没有「第 1 步、第 2 步」**。

### 1.2 序列任务长什么样？

例子：用前 3 天天气预测第 4 天是否下雨。

| 天 | 温度 | 气压 | 是否下雨 |
|----|------|------|----------|
| 第1天 | … | … | … |
| 第2天 | … | … | … |
| 第3天 | … | … | … |

如果把 3 天拼成一个长向量塞进 Linear：

- 能算，但**「第 2 天在第 1 天后面」**这种顺序信息被抹平了  
- 若改成「用 7 天预测」，网络输入维要改，规则也不能自然复用  

RNN 的做法：

```
第1天特征 x1 + h0 → Cell → h1
第2天特征 x2 + h1 → Cell → h2
第3天特征 x3 + h2 → Cell → h3
再用 h3 做预测
```

**同一套 Cell 权重**扫完整段序列；记忆靠 `h` 传递。

### 1.3 Java 类比

```java
// 伪代码：不是真 API
Hidden h = zeros();
for (Feature x : sequence) {
    h = cell.forward(x, h);  // 同一个 cell，循环调用
}
```

像用同一个 `Service` 对象处理队列里的每一条消息，还带着上一次的 `state`。

---

## 2. 五个参数分别是什么？（先背名字）

后面所有 shape 都围着这 5 个转：

| 参数 | 英文 | 含义 | hello 例子里的值 |
|------|------|------|------------------|
| **序列长度** | `seq_len` | 有多少个时间步 | 5（h,e,l,l,o 五个字符） |
| **批量大小** | `batch_size` | 一次并行算几条序列 | 课件常用 1 |
| **输入特征维** | `input_size` | **每一个时间步**的向量有多长 | 4（One-Hot：e/h/l/o） |
| **隐藏维** | `hidden_size` | 记忆向量 `h` 有多长 | 课件常用 4 |
| **层数** | `num_layers` | 纵向叠几层 RNN（先当 1） | 1 |

### 易混点（必读）

1. **`input_size` 不是「整段序列的长度」**  
   - 序列长度 = `seq_len`  
   - `input_size` = **单个时间步**的特征维  

2. **`hidden_size` 不是类别数**（虽然玩具例子里碰巧都是 4）  
   - 隐藏维是你设计的「记忆容量」  
   - 分类类别数可以另接 Linear；课件把 `hidden_size` 直接当 4 类 logits 用，是简化  

3. **`batch_size` 是「几条序列一起算」**，不是「几个时间步」  

---

## 3. 三维张量怎么读？（seq, batch, feature）

PyTorch 默认 RNN 输入是：

```
(seq_len, batch_size, input_size)
```

用「书架」想象：

```
第 0 维 seq_len：时间往前翻页（第几个字符）
第 1 维 batch：   同一时刻并排放几条样本
第 2 维 feature： 这一步的向量内容
```

### 3.1 只有 1 条序列时（batch=1）

`hello` One-Hot 后：

```
shape = (5, 1, 4)

时间 t=0 (字符 h):  [[0,1,0,0]]     ← shape (1, 4)，外面还有一层 batch
时间 t=1 (字符 e):  [[1,0,0,0]]
时间 t=2 (字符 l):  [[0,0,1,0]]
时间 t=3 (字符 l):  [[0,0,1,0]]
时间 t=4 (字符 o):  [[0,0,0,1]]
```

读法：

- `inputs[0]` → 第 0 步，shape `(1, 4)` = `(batch, input_size)`  
- `inputs[0, 0]` → 第 0 步、第 0 条样本，shape `(4,)`  

### 3.2 若 batch=2（两条序列一起）

```
shape = (5, 2, 4)
inputs[t, 0, :]  → 第 1 条序列在时刻 t 的向量
inputs[t, 1, :]  → 第 2 条序列在时刻 t 的向量
```

两条序列**各自**维护自己的 hidden（batch 维上并列），互不干扰。

### 3.3 `batch_first=True` 时

输入变成：

```
(batch_size, seq_len, input_size)
```

只是**前两维对调**，内容一样。  
注意：**h0 的 shape 不跟着变成 batch 在前**，仍是 `(num_layers, batch, hidden_size)`。

---

## 4. 隐藏状态 h：形状与含义

### 4.1 对 RNNCell

| 张量 | shape |
|------|-------|
| 当前步输入 `x_t` | `(batch, input_size)` |
| 当前记忆 `h` | `(batch, hidden_size)` |
| Cell 输出（新 h） | `(batch, hidden_size)` |

一步：

```python
h = cell(x_t, h)   # 用新 h 覆盖旧 h，带去下一步
```

### 4.2 对 nn.RNN（整段）

| 张量 | shape |
|------|-------|
| 输入 `inputs` | `(seq_len, batch, input_size)` |
| 初始 `h0` | `(num_layers, batch, hidden_size)` |
| 输出 `out` | `(seq_len, batch, hidden_size)` ← **每一步**的 h |
| 最终 `hidden` | `(num_layers, batch, hidden_size)` ← **最后一步**的 h |

关系：

- `out[t]` ≈ 第 t 步的隐藏状态  
- `hidden[-1]`（单层时就是 `hidden[0]`）≈ 最后一步的 h  
- 单层时：`out[-1]` 与最终 `hidden` 对应（数值上一致/同语义）

### 4.3 为什么要 h0？

第一天没有「昨天」，必须给一个起点：

- 常见：全 0  
- 特殊：用别的网络产物当 h0（如图像→文字）

**每开始一条新序列，都要重新初始化 h0**，否则会把上一条序列的记忆带过来。

---

## 5. RNNCell 逐步拆解（建议先跑通）

### 5.1 构造

```python
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
```

含义：

- 每一步吃长度 4 的向量  
- 吐出长度 2 的记忆  
- **没有** `seq_len` 参数：序列多长由你自己 for 几圈决定  

### 5.2 假数据 + 循环

```python
# (seq_len, batch, input_size)
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

for t in range(seq_len):
    x_t = dataset[t]                 # (batch, input_size) = (1, 4)
    print(f"t={t} x_t:", x_t.shape)
    hidden = cell(x_t, hidden)       # (batch, hidden_size) = (1, 2)
    print(f"t={t} h  :", hidden.shape)
```

你应看到：

```
t=0 x_t: torch.Size([1, 4])
t=0 h  : torch.Size([1, 2])
t=1 ...
```

### 5.3 这一步在算什么（直觉）

```
h_new = tanh( W_ih @ x_t  +  W_hh @ h_old  + bias )
```

- `W_ih`：把「当前输入」映到 hidden 空间  
- `W_hh`：把「旧记忆」映到 hidden 空间  
- 相加再 tanh：融合「现在」和「过去」  

**同一组 W，每个 t 都用一遍** → 权值共享。

---

## 6. nn.RNN 一次吃整段（对照 Cell）

```python
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

rnn = torch.nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
)

inputs = torch.randn(seq_len, batch_size, input_size)     # (3,1,4)
h0 = torch.zeros(num_layers, batch_size, hidden_size)     # (1,1,2)

out, hidden = rnn(inputs, h0)

print("inputs:", inputs.shape)   # (3, 1, 4)
print("h0    :", h0.shape)       # (1, 1, 2)
print("out   :", out.shape)      # (3, 1, 2)  三步各有一个 h
print("hidden:", hidden.shape)   # (1, 1, 2)  最后一步
```

### 和 Cell 的对应关系

| RNNCell 手写 | nn.RNN |
|--------------|--------|
| `for t in range(seq_len): h=cell(x[t],h)` | 内部自动 for |
| 你只拿得到「当前 h」 | `out` 里保存了每一步的 h |
| h 形状 `(batch, hidden)` | h0/hidden 多一维 `num_layers` |

单层时可以记：

```
nn.RNN ≈ for 循环版 RNNCell + 把每步 h 叠成 out
```

---

## 7. hello → ohlol：把任务钉死

### 7.1 任务在干什么

```
输入字符序列:  h e l l o
目标字符序列:  o h l o l
```

不是「整句一个标签」，而是：

> **每个时间步做一个 4 分类**：这一步输出对应哪个字符。

| t | 输入 | 目标 |
|---|------|------|
| 0 | h | o |
| 1 | e | h |
| 2 | l | l |
| 3 | l | o |
| 4 | o | l |

### 7.2 字典与下标

```python
idx2char = ['e', 'h', 'l', 'o']
# 下标:      0    1    2    3

x_data = [1, 0, 2, 2, 3]  # h e l l o
y_data = [3, 1, 2, 3, 2]  # o h l o l
```

### 7.3 One-Hot：字符 → 向量

神经网络吃数字向量，不吃字母。

```python
# 类别数 = input_size = 4
one_hot_lookup = [
    [1, 0, 0, 0],  # e
    [0, 1, 0, 0],  # h
    [0, 0, 1, 0],  # l
    [0, 0, 0, 1],  # o
]

x_one_hot = [one_hot_lookup[i] for i in x_data]
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
# → (5, 1, 4)

labels = torch.LongTensor(y_data)           # RNN 版常用 (5,)
# 或 Cell 版逐步: labels.view(seq_len, 1)  # 每步 (1,)
```

**为什么 labels 是 Long，不是 One-Hot？**  
`CrossEntropyLoss` 要的是**类别下标**（0～3），不是 one-hot 向量。  
输入用 one-hot；标签用整数下标。

### 7.4 本课的简化：hidden 直接当 logits

课件常设：

```python
input_size = 4
hidden_size = 4   # 碰巧 = 类别数
```

于是每步的 `h` 直接当成 4 个类别的分数，接 CE。  

更「正规」的做法是：`hidden_size` 可以是 8/16/32，再接 `Linear(hidden, 4)`。  
**先跟课件的简化版把维度跑通，再理解「可以加 Linear」。**

---

## 8. 两种实现对照（hello 任务）

### 8.1 RNNCell 版：一步一步算 loss

```python
# 伪结构（维度注释是重点）
hidden = net.init_hidden()          # (batch, hidden) = (1, 4)
total_loss = 0

for t in range(seq_len):
    x_t = inputs[t]                 # (1, 4)
    y_t = labels[t]                 # 标量或 (1,) 的 Long
    hidden = net(x_t, hidden)       # (1, 4) 当作 logits
    total_loss = total_loss + criterion(hidden, y_t)

total_loss.backward()
optimizer.step()
```

要点：

- **一个 epoch 内**，5 步共用一条计算图，loss 累加后再 `backward`（课件写法）  
- 预测：`_, idx = hidden.max(dim=1)` → `idx2char[idx]`  

### 8.2 nn.RNN 版：一次出全部时间步

```python
out = net(inputs)
# 若 forward 里: out, _ = rnn(inputs, h0); return out.view(-1, hidden_size)
# out_for_ce: (seq_len * batch, hidden) = (5, 4)
# labels:     (5,)

loss = criterion(out_for_ce, labels)
loss.backward()
```

要点：

- 不再手写 for（RNN 内部已循环）  
- `view(-1, hidden_size)` 是为了让 CE 吃 `(N, C)` 和 `(N,)`  

### 8.3 预测字符串

```python
_, idx = outputs.max(dim=1)   # 每行最大分数的列号
chars = [idx2char[i] for i in idx.tolist()]
print(''.join(chars))         # 期望逐渐变成 ohlol
```

---

## 9. 参数清单：创建时 vs 运行时

### 9.1 创建 `RNNCell` / `RNN` 时你要传的

```python
nn.RNNCell(input_size, hidden_size)
nn.RNN(input_size, hidden_size, num_layers=1, batch_first=False)
```

| 参数 | 决定什么 |
|------|----------|
| `input_size` | 权重 `W_ih` 的「输入宽度」 |
| `hidden_size` | `h` 的长度；也是输出特征维 |
| `num_layers` | 纵向堆叠层数（Cell 没有这个，RNN 有） |
| `batch_first` | 只影响你喂数据的维顺序 |

**创建时不需要 `seq_len`、`batch_size`。**  
序列多长、batch 多大，是**喂数据时**的事（Cell 对 batch 维自适应；h0 要自己按 batch 建）。

### 9.2 运行时你要准备的

| 你要准备 | Cell | RNN |
|----------|------|-----|
| `x` / `inputs` | 每步 `(B, input_size)` | 整段 `(S, B, input_size)` |
| `h` / `h0` | `(B, hidden)` | `(num_layers, B, hidden)` |
| `labels` | 每步类别 | `(S*B,)` 或逐步 |

---

## 10. 一张表串起所有 shape（默认真经）

设：`S=seq_len, B=batch, I=input_size, H=hidden_size, L=num_layers`

### RNNCell

| 变量 | shape |
|------|-------|
| 一步输入 | `(B, I)` |
| h | `(B, H)` |
| 一步输出 h | `(B, H)` |

### nn.RNN（batch_first=False）

| 变量 | shape |
|------|-------|
| inputs | `(S, B, I)` |
| h0 | `(L, B, H)` |
| out | `(S, B, H)` |
| hidden | `(L, B, H)` |

### hello 玩具（S=5,B=1,I=4,H=4,L=1）

| 变量 | shape |
|------|-------|
| inputs | `(5, 1, 4)` |
| h0 | `(1, 1, 4)` |
| out | `(5, 1, 4)` |
| view 后接 CE | `(5, 4)` |
| labels | `(5,)` |

把这张表抄在纸上，听课/写代码时对照，懵的概率会大幅下降。

---

## 11. 训练循环在干什么（和之前课对齐）

和线性回归 / MNIST **同构**，只是「一个 batch」变成「一段序列」：

```
1. 准备 inputs / labels（One-Hot + Long）
2. 清梯度 optimizer.zero_grad()
3. 前向：Cell 循环 或 RNN 一次
4. 算 CE loss
5. loss.backward()
6. optimizer.step()
7. （可选）用 max 打印当前预测字符串
```

新知识点只有：

- 数据多了时间维  
- 前向里多了 hidden 传递  
- 每条序列前要 init hidden  

---

## 12. 建议自学顺序（吃透用）

按顺序做，做完一项打勾：

### Day 片段 A：只碰维度（30 分钟）

- [ ] 跑通 §5 RNNCell 随机数据，看懂每步 shape  
- [ ] 跑通 §6 nn.RNN，对比 `out` 与 `hidden`  
- [ ] 改 `batch_size=2`，看哪些维变成 2  

### Day 片段 B：One-Hot（20 分钟）

- [ ] 手写 `hello` 的 5 个 one-hot  
- [ ] `print(inputs.shape)` 确认为 `(5,1,4)`  
- [ ] `print(labels.dtype)` 确认为整型  

### Day 片段 C：Cell 版 hello（45 分钟）

- [ ] 跟课件敲 RNNCell 模型  
- [ ] 训练时打印每轮预测字符串  
- [ ] 观察是否逐渐接近 `ohlol`  

### Day 片段 D：RNN 版 hello（45 分钟）

- [ ] 改成 `nn.RNN`，去掉手写 for  
- [ ] 搞清 `view(-1, hidden_size)` 为什么  
- [ ] 与 Cell 版对比：哪里少了循环、shape 哪里不同  

### Day 片段 E：变体（巩固）

- [ ] `batch_first=True`，改 inputs 排列，仍训通  
- [ ] 把 `hidden_size` 改成 8，后面加 `nn.Linear(8, 4)` 再 CE（理解「隐藏维 ≠ 类别数」）  

---

## 13. 自问自答（用来查漏）

**Q1：`input_size=4` 是因为字符串长度是 4 吗？**  
A：不是。是因为字典里有 4 种字符（One-Hot 长度）。字符串长度是 `seq_len=5`。

**Q2：为什么 h0 是 `(num_layers, batch, hidden)`，多一层？**  
A：多层 RNN 时每一层都有自己的最终状态；单层时第一维就是 1。

**Q3：`out` 和 `hidden` 都要吗？**  
A：逐步分类常用整段 `out`；只要句末向量时用最后的 `hidden`（或 `out[-1]`）。hello 任务用每步输出 → 用 `out`。

**Q4：可以不 One-Hot 吗？**  
A：可以，改用 `nn.Embedding`（第 13 课/工程更常见）。本课先掌握 One-Hot 与维度即可。

**Q5：loss 为什么有时对每步累加？**  
A：每个时间步都是一次分类，总损失是各步 CE 之和（或平均），整段一起反传，让 Cell 权重学会整串规律。

**Q6：和 CNN 的「通道」是一回事吗？**  
A：不是。CNN 的 C 是特征图通道；RNN 的 `input_size`/`hidden_size` 是**每个时间步的向量长度**。别用 NCHW 硬套。

---

## 14. 最短记忆卡片（背这 8 行）

1. RNN = 共享权重的 Cell + 时间上的 for  
2. `seq_len` = 几步；`input_size` = 每步向量多长  
3. `hidden_size` = 记忆向量多长  
4. 默认输入 `(S, B, I)`  
5. Cell：`(B,I)+(B,H)→(B,H)`  
6. RNN：`(S,B,I)+h0→out(S,B,H)`  
7. 字符：One-Hot 作输入，Long 下标作标签  
8. 新序列必须重置 h0  

---

## 15. 推荐最小可运行实验（复制即跑）

把下面存成 `homework/day08/rnn_shape_demo.py` 之类，先跑通再写 hello：

```python
import torch

print("===== RNNCell =====")
cell = torch.nn.RNNCell(input_size=4, hidden_size=2)
x_seq = torch.randn(5, 1, 4)          # S,B,I
h = torch.zeros(1, 2)                 # B,H
for t in range(5):
    h = cell(x_seq[t], h)
    print(t, "x", x_seq[t].shape, "h", h.shape)

print("===== RNN =====")
rnn = torch.nn.RNN(input_size=4, hidden_size=2, num_layers=1)
x = torch.randn(5, 1, 4)
h0 = torch.zeros(1, 1, 2)             # L,B,H
out, hn = rnn(x, h0)
print("out", out.shape, "hn", hn.shape)
```

期望：

```
Cell: 每步 x [1,4], h [1,2]
RNN:  out [5,1,2], hn [1,1,2]
```

---

## 16. 和总结文档怎么配合用

| 文档 | 用途 |
|------|------|
| `第12课-循环神经网络基础-总结.md` | 复习提纲、自测清单 |
| **本文** | 维度懵时查阅、按 §12 顺序动手 |

建议路径：本文 §5→§6→§7→§8 跟做一遍 → 再回总结做自测清单。

---

*若某一段仍懵：告诉我是「§几 + 哪张表 / 哪行代码」，可以只针对那一段再拆一版（带你当前打印出来的 shape）。*
