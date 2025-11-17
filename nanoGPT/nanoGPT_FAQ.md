# nanoGPT 学习笔记 - FAQ

本文档整理了学习 nanoGPT 过程中的常见问题和详细解答。

---

## 目录

1. [字符级 Tokenization 详解](#1-字符级tokenization详解)
2. [enumerate()函数](#2-enumerate函数)
3. [模型参数追踪机制](#3-模型参数追踪机制)
4. [为什么模型输出是随机的](#4-为什么模型输出是随机的)
5. [loss.backward()的作用](#5-lossbackward的作用)

---

## 1. 字符级 Tokenization 详解

### 问题

字符级 tokenization 中的 `stoi`、`itos`、`encode`、`decode` 分别做了什么？单个字符没有像单词那样的上下文意义，为什么要用字符级 tokenization？

### 代码示例

```python
# 所有不同的字符
chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?',
         'A', 'B', 'C', ..., 'Z', 'a', 'b', 'c', ..., 'z']  # 共65个

# 字符级tokenization
stoi = { ch:i for i,ch in enumerate(chars) }  # string to index
itos = { i:ch for i,ch in enumerate(chars) }  # index to string

# encode和decode
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 使用示例
test_string = 'hello world'
encoded = encode(test_string)  # [46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42]
decoded = decode(encoded)      # 'hello world'
```

### 详细解释

#### 1.1 `stoi` (string to index)

```python
stoi = { ch:i for i,ch in enumerate(chars) }
```

创建一个**字符 → 数字**的查询字典：

```python
{
    '\n': 0,
    ' ': 1,
    '!': 2,
    ...
    'h': 46,
    'e': 43,
    'l': 50,
    'o': 53
}
```

**用途**：输入字符，查询对应的数字编号

```python
stoi['h']  # 返回 46
stoi['e']  # 返回 43
```

#### 1.2 `itos` (index to string)

```python
itos = { i:ch for i,ch in enumerate(chars) }
```

创建一个**数字 → 字符**的反向字典：

```python
{
    0: '\n',
    1: ' ',
    2: '!',
    ...
    46: 'h',
    43: 'e',
    50: 'l',
    53: 'o'
}
```

**用途**：输入数字，查询对应的字符

```python
itos[46]  # 返回 'h'
itos[43]  # 返回 'e'
```

#### 1.3 `encode` 函数

```python
encode = lambda s: [stoi[c] for c in s]
```

把**文本转换成数字列表**：

```python
encode('hello')
# 'h' → 46
# 'e' → 43
# 'l' → 50
# 'l' → 50
# 'o' → 53
# 返回: [46, 43, 50, 50, 53]
```

**为什么需要？** 神经网络只能处理数字，不能直接处理文本。

#### 1.4 `decode` 函数

```python
decode = lambda l: ''.join([itos[i] for i in l])
```

把**数字列表转换回文本**：

```python
decode([46, 43, 50, 50, 53])
# 46 → 'h'
# 43 → 'e'
# 50 → 'l'
# 50 → 'l'
# 53 → 'o'
# 拼接成: 'hello'
```

**为什么需要？** 模型生成的是数字，我们需要转回人类可读的文字。

### 为什么使用字符级 Tokenization？

#### 关键点：这不是最终的表示！

字符级 tokenization 只是输入层的编码方式：

```
字符 → 数字 → Embedding层 → Transformer层（学习上下文）
```

模型会通过**多层神经网络学习**字符之间的关系和模式。

#### 字符级 tokenization 的优势

| 优势              | 说明                                     |
| ----------------- | ---------------------------------------- |
| ✅ 词汇表很小     | 只有 65 个字符 vs 单词级可能有几万个单词 |
| ✅ 没有未知词问题 | 任何文本都能用这 65 个字符表示           |
| ✅ 可学习拼写规则 | 能学习 walk/walking/walked 的形态变化    |
| ✅ 适合小数据集   | 不需要庞大的词汇表                       |

#### 模型如何学习"上下文意义"

```
输入字符: h e l l o

经过Transformer后，模型学会：
- "h" 后面经常跟 "e", "i", "a"
- "qu" 经常一起出现
- "ing" 是常见结尾
- 整个单词 "hello" 的含义
```

**类比**：

- **字符级**：就像学字母 → 拼单词 → 理解句子（模型自己学会拼写和语法）
- **单词级**：直接给单词 → 理解句子（需要预定义词汇表，无法处理新词）

#### 完整流程

```
原始文本 ──encode──> 数字 ──神经网络处理──> 输出数字 ──decode──> 生成文本
"hello"              [46,43,50,50,53]    [...]                "world"
```

模型**不是孤立地看每个字符**，而是通过**注意力机制（Attention）**看到字符的**组合和序列模式**，从而理解更高层次的含义！

---

## 2. enumerate()函数

### 问题

`enumerate()` 这个函数是干什么的？

### 基本语法

```python
enumerate(iterable, start=0)
```

**作用**：同时获取**索引（index）**和**值（value）**

### 简单例子

```python
chars = ['a', 'b', 'c', 'd']

# 不用enumerate（麻烦）
for i in range(len(chars)):
    print(i, chars[i])

# 用enumerate（优雅）
for i, ch in enumerate(chars):
    print(i, ch)

# 两者输出相同:
# 0 a
# 1 b
# 2 c
# 3 d
```

### enumerate 的工作原理

```python
chars = ['a', 'b', 'c', 'd']

# enumerate(chars) 产生:
# (0, 'a')
# (1, 'b')
# (2, 'c')
# (3, 'd')
```

### 在 tokenization 代码中的应用

```python
chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ...]

# enumerate(chars) 会产生:
# (0, '\n')
# (1, ' ')
# (2, '!')
# (3, '$')
# (4, '&')
# ...

# 创建 stoi (string to index)
stoi = { ch:i for i,ch in enumerate(chars) }
# 遍历时：
# i=0, ch='\n'  → {'\n': 0}
# i=1, ch=' '   → {' ': 1}
# i=2, ch='!'   → {'!': 2}
# 最终结果: {'\n': 0, ' ': 1, '!': 2, ...}

# 创建 itos (index to string)
itos = { i:ch for i,ch in enumerate(chars) }
# 遍历时：
# i=0, ch='\n'  → {0: '\n'}
# i=1, ch=' '   → {1: ' '}
# i=2, ch='!'   → {2: '!'}
# 最终结果: {0: '\n', 1: ' ', 2: '!', ...}
```

### 对比三种方法

```python
chars = ['a', 'b', 'c']

# 方法1：手动计数（不推荐）
i = 0
for ch in chars:
    print(f"索引{i}: {ch}")
    i += 1

# 方法2：用range和索引（啰嗦）
for i in range(len(chars)):
    print(f"索引{i}: {chars[i]}")

# 方法3：用enumerate（最佳！）
for i, ch in enumerate(chars):
    print(f"索引{i}: {ch}")

# 三种方法输出都是：
# 索引0: a
# 索引1: b
# 索引2: c
```

### 实际应用例子

```python
# 在文本中找到某个字符的所有位置
text = "hello"
for index, char in enumerate(text):
    if char == 'l':
        print(f"字母'l'在位置{index}")

# 输出:
# 字母'l'在位置2
# 字母'l'在位置3
```

### enumerate 的可选参数

```python
# 可以指定起始索引（默认是0）
for i, ch in enumerate(['a', 'b', 'c'], start=1):
    print(i, ch)

# 输出:
# 1 a
# 2 b
# 3 c
```

**总结**：`enumerate()` 就是一个**计数器**，让你在遍历列表时不用手动维护索引，非常方便！

---

## 3. 模型参数追踪机制

### 问题

`optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)` 中，为什么 `parameters()` 就可以获取到所有需要训练的参数？是所有参与了 forward 计算的变量都会被追踪然后更新吗？

### 核心答案

**并不是所有参与 forward 的变量都会被追踪！**

`model.parameters()` 只会返回**显式注册为可训练参数的变量**，而不是所有参与计算的变量。

### 什么会被自动注册为参数？

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # ✅ 这会被自动注册为参数
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

**关键点：**

- 当你在 `__init__` 中把 `nn.Module` 的子类（如 `nn.Embedding`, `nn.Linear`, `nn.Conv2d` 等）赋值给 `self.xxx` 时
- PyTorch 会**自动检测**并注册这些层的所有参数
- `nn.Embedding(vocab_size, vocab_size)` 内部有一个权重矩阵，这个矩阵会被自动注册

### 不同类型变量的对比

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ 会被注册为参数（通过 nn.Parameter）
        self.weight1 = nn.Parameter(torch.randn(10, 10))

        # ✅ 会被注册为参数（nn.Linear 内部有参数）
        self.linear = nn.Linear(10, 10)

        # ❌ 不会被注册（普通tensor）
        self.temp_tensor = torch.randn(10, 10)

        # ❌ 不会被注册（普通Python变量）
        self.scale_factor = 2.0

        # ✅ 注册为buffer（不需要梯度，但会被保存）
        self.register_buffer('running_mean', torch.zeros(10))

    def forward(self, x):
        # 所有这些都参与计算，但只有注册的参数会被优化器更新！
        x = x * self.scale_factor  # scale_factor 不会被优化
        x = x + self.temp_tensor   # temp_tensor 不会被优化
        x = x @ self.weight1       # weight1 会被优化 ✅
        x = self.linear(x)         # linear的权重会被优化 ✅
        return x

model = MyModel()

# 查看所有参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 输出:
# weight1: torch.Size([10, 10])
# linear.weight: torch.Size([10, 10])
# linear.bias: torch.Size([10])
```

### BigramLanguageModel 中的参数

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # nn.Embedding 创建了一个 (vocab_size, vocab_size) 的权重矩阵
        # 这个矩阵被自动注册为参数！
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # logits: 从embedding表中查找
        logits = self.token_embedding_table(idx)
        # ... 其他计算
```

**查看参数：**

```python
model = BigramLanguageModel(vocab_size=65)

# 方法1：查看所有参数
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

# 输出:
# token_embedding_table.weight: shape=torch.Size([65, 65]), requires_grad=True

# 方法2：计算参数总数
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# 输出: Total parameters: 4225 (也就是 65*65)
```

### nn.Module 的魔法原理

```python
class nn.Module:
    def __setattr__(self, name, value):
        # 当你写 self.xxx = yyy 时，这个方法会被调用

        if isinstance(value, nn.Parameter):
            # 如果是 Parameter，注册到 _parameters 字典
            self._parameters[name] = value

        elif isinstance(value, nn.Module):
            # 如果是 Module（如 nn.Linear），注册到 _modules 字典
            # 并递归收集它的所有参数
            self._modules[name] = value

        else:
            # 普通变量，不注册
            object.__setattr__(self, name, value)

    def parameters(self):
        # 递归收集所有 _parameters 和子模块的 parameters
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
```

### 实际验证

```python
import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.randn(5, 5))  # ✅ 会被追踪
        self.tensor1 = torch.randn(5, 5)               # ❌ 不会被追踪

    def forward(self, x):
        # 两个都参与计算！
        return x + self.param1 + self.tensor1

model = TestModel()

print("Parameters:")
for name, p in model.named_parameters():
    print(f"  {name}")

# 输出:
# Parameters:
#   param1

# tensor1 虽然参与了forward，但不在parameters()中！
```

### 总结对比表

| **变量类型**             | **是否被`parameters()`追踪** | **是否参与 forward** | **是否被优化器更新** |
| ------------------------ | ---------------------------- | -------------------- | -------------------- |
| `nn.Parameter`           | ✅ 是                        | 可以                 | ✅ 是                |
| `nn.Linear/Embedding等`  | ✅ 是（递归）                | 可以                 | ✅ 是                |
| 普通`torch.Tensor`       | ❌ 否                        | 可以                 | ❌ 否                |
| Python 变量（int/float） | ❌ 否                        | 可以                 | ❌ 否                |
| `register_buffer`        | ❌ 否（但会被保存）          | 可以                 | ❌ 否                |

### 关键规则

1. **只有显式注册的参数才会被优化**
2. **参与 forward 计算 ≠ 会被优化器更新**
3. **`nn.Module` 通过 Python 的 `__setattr__` 魔法方法自动检测参数**
4. **`model.parameters()` 递归收集所有注册的参数**

在 BigramLanguageModel 中：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# 只会优化 token_embedding_table.weight 这一个参数矩阵！
```

---

## 4. 为什么模型输出是随机的

### 问题

如果模型只是看当前字符预测下一个，比如 "e" 后面是 "r"，那应该是固定的呀，为什么每次生成的文本都不一样？

### 核心答案

**误区**：认为 "e 后面必然是 r" → 输出应该固定

**真相**：模型输出的是**概率分布**，然后**随机采样**！

### 模型输出的是什么？

```python
# 假设当前字符是 'e' (编号43)
logits = self.token_embedding_table(idx)  # 查找embedding表
# logits 是一个长度65的向量，每个位置对应一个字符的"得分"

# 例如 logits 可能是:
# [0.1, 0.3, -0.5, ..., 2.1(r的位置), ..., 1.5(其他字符), ...]

# 转成概率分布
probs = F.softmax(logits, dim=-1)
# probs 现在是概率，加起来等于1:
# [0.01, 0.02, 0.005, ..., 0.35(r), 0.15(d), 0.10(s), 0.05(n), ...]
```

**关键点：不是只有一个答案！**

### 训练数据中"e"后面有很多可能

在莎士比亚的文本中，"e" 后面可能跟：

- "e" → "r" (hear, dear, fear)
- "e" → "d" (need, feed, seed)
- "e" → " " (the end, here on, were all)
- "e" → "s" (comes, takes, makes)
- "e" → "n" (open, even, seven)

**模型学到的是所有这些可能性的分布！**

```python
# 假设在训练数据中统计:
# "e" 后面跟 "r": 30% 的时间
# "e" 后面跟 "d": 15% 的时间
# "e" 后面跟 " ": 20% 的时间
# "e" 后面跟 "s": 10% 的时间
# "e" 后面跟其他: 25% 的时间

# 模型学到的概率分布就会接近这个统计！
probs = [
    ...,
    0.30,  # 'r' 的概率
    0.15,  # 'd' 的概率
    0.20,  # ' ' 的概率
    0.10,  # 's' 的概率
    ...
]
```

### torch.multinomial 是随机采样

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = self(idx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # 关键：按概率随机选择，不是选最大的！
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**`torch.multinomial` 是关键！** 这个函数**按概率随机选择**。

#### 随机采样例子

```python
import torch

probs = torch.tensor([0.30, 0.15, 0.20, 0.10, 0.25])  # 5个字符的概率

# 采样10次，看结果：
for i in range(10):
    sample = torch.multinomial(probs, num_samples=1)
    print(f"采样{i+1}: 选择了字符 {sample.item()}")

# 可能的输出（每次运行都不同！）:
# 采样1: 选择了字符 0 (30%概率)
# 采样2: 选择了字符 4 (25%概率)
# 采样3: 选择了字符 2 (20%概率)
# 采样4: 选择了字符 0 (30%概率)
# 采样5: 选择了字符 1 (15%概率)
# 采样6: 选择了字符 2 (20%概率)
# ...
```

**每次采样都可能不同，但长期来看会符合概率分布！**

### 对比：随机采样 vs 贪婪采样

```python
# 方法1：随机采样（代码中使用的）
idx_next = torch.multinomial(probs, num_samples=1)
# 结果：多样化，每次运行不同 ✅

# 方法2：贪婪采样（总是选最大概率的）
idx_next = torch.argmax(probs, dim=-1, keepdim=True)
# 结果：确定性，每次运行相同 ❌ 但很无聊，可能陷入循环
```

### 为什么用随机采样？

| 优势            | 说明                       |
| --------------- | -------------------------- |
| ✅ 增加多样性   | 生成的文本更有趣           |
| ✅ 避免重复     | 不会一直生成 "erererer..." |
| ✅ 探索更多可能 | 低概率的词也有机会出现     |

### 实验验证

```python
# 生成3次，看看结果是否不同
for i in range(3):
    output = model.generate(
        idx=torch.zeros((1, 1), dtype=torch.long),
        max_new_tokens=50
    )
    print(f"第{i+1}次生成:")
    print(decode(output[0].tolist()))
    print("-" * 50)

# 你会看到3次输出完全不同！
```

### 如果想要确定性输出

```python
# 修改generate函数，使用argmax而不是multinomial
def generate_deterministic(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = self(idx)
        logits = logits[:, -1, :]
        # 总是选最大的
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# 这样每次生成的结果就会相同了！
# 但文本质量通常会下降，因为太单调
```

### 总结对比表

| **理解**             | **实际情况**                                        |
| -------------------- | --------------------------------------------------- |
| "e"后面必然跟"r"     | "e"后面有**概率分布**：r(30%), d(15%), 空格(20%)... |
| 输出应该固定         | 使用**随机采样**，每次按概率选择                    |
| 模型应该输出确定答案 | 模型输出**概率**，采样产生最终字符                  |

### 关键点

- **模型输出** = 概率分布（所有可能的概率）
- **`torch.multinomial`** = 按概率随机选择
- **每次运行** = 不同的采样结果
- **这是特性不是 Bug**，增加了生成的多样性！

如果你想要确定性输出，改用 `torch.argmax` 就行，但生成的文本会更单调。

---

## 5. loss.backward()的作用

### 问题

训练循环中的 `loss.backward()` 在这里的作用是什么？

### 核心答案

**`loss.backward()` 的作用：计算所有参数的梯度**

`loss.backward()` 执行**反向传播（Backpropagation）**，计算损失函数对模型所有参数的梯度（导数）。

### 完整训练循环

```python
for steps in range(10000):
    # 步骤1: 获取数据
    xb, yb = get_batch('train')

    # 步骤2: 前向传播 - 计算预测和损失
    logits, loss = model(xb, yb)

    # 步骤3: 清空之前的梯度
    optimizer.zero_grad(set_to_none=True)

    # 步骤4: 反向传播 - 计算梯度 ⭐
    loss.backward()

    # 步骤5: 更新参数
    optimizer.step()
```

### 逐步详解

#### 步骤 1: 获取数据

```python
xb, yb = get_batch('train')
# xb: 输入数据 (batch_size, block_size)
# yb: 目标数据 (batch_size, block_size)
```

#### 步骤 2: 前向传播

```python
logits, loss = model(xb, yb)
# logits: 模型预测 (batch_size, block_size, vocab_size)
# loss: 标量，表示预测与真实值的差距
```

**这时发生了什么？**

- PyTorch **自动构建计算图（Computation Graph）**
- 记录所有操作和参数
- 为反向传播做准备

```python
# 简化的计算图：
# xb → embedding → logits → cross_entropy → loss
#      ↑ (参数)
#   weights
```

#### 步骤 3: 清空梯度

```python
optimizer.zero_grad(set_to_none=True)
```

**为什么需要？**

- PyTorch 的梯度是**累加**的
- 上一次迭代的梯度还在
- 必须清零，否则新旧梯度会叠加

```python
# 清零前：param.grad = [上次的梯度]
optimizer.zero_grad()
# 清零后：param.grad = None (或 0)
```

#### 步骤 4: 反向传播 ⭐ 核心

```python
loss.backward()
```

**这一步做了什么？**

1. **沿着计算图反向传播**
2. **计算 loss 对每个参数的梯度**（偏导数）
3. **把梯度存储在 `param.grad` 中**

```python
# 反向传播过程（简化）：
# loss (标量)
#   ↓ 反向传播
# ∂loss/∂logits
#   ↓ 反向传播
# ∂loss/∂embedding_weights  ← 存储在 weights.grad
```

**数学上：**

```
假设 loss = f(w1, w2, w3, ...)
其中 w1, w2, w3 是模型参数

loss.backward() 计算：
- ∂loss/∂w1 → 存储在 w1.grad
- ∂loss/∂w2 → 存储在 w2.grad
- ∂loss/∂w3 → 存储在 w3.grad
...
```

#### 步骤 5: 更新参数

```python
optimizer.step()
```

**使用计算好的梯度更新参数：**

```python
# 简化版（实际AdamW更复杂）
for param in model.parameters():
    param.data = param.data - learning_rate * param.grad
```

### 形象比喻

把模型训练想象成**爬山找最低点**：

```
步骤1-2: 站在山上某个位置（当前参数），看看周围有多高（计算loss）

步骤3: 擦掉上次的脚印标记（清空梯度）

步骤4: loss.backward()
       → 用指南针测量四周，哪个方向最陡（计算梯度）
       → 在地图上标记方向（存储梯度）

步骤5: optimizer.step()
       → 沿着最陡的方向走一小步（更新参数）
```

### 实际代码验证

```python
import torch
import torch.nn as nn

# 简单模型
model = nn.Linear(2, 1)  # 2个输入，1个输出
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 查看初始参数
print("初始权重:", model.weight.data)
print("初始偏置:", model.bias.data)
print("初始梯度:", model.weight.grad)  # None

# 训练一步
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[3.0]])

# 前向传播
pred = model(x)
loss = (pred - y) ** 2

print("\n前向传播后:")
print("预测:", pred.item())
print("损失:", loss.item())
print("梯度还是:", model.weight.grad)  # 还是 None

# 反向传播 ⭐
loss.backward()

print("\nbackward()后:")
print("梯度出现了!", model.weight.grad)  # 不再是 None！
# 输出类似: tensor([[0.5234, 1.0468]])

# 更新参数
optimizer.step()

print("\nstep()后:")
print("新权重:", model.weight.data)  # 权重改变了！
```

### 完整流程图

```
┌─────────────────────────────────────────────────┐
│          前向传播 (Forward Pass)                 │
├─────────────────────────────────────────────────┤
│ input → embedding → logits → loss               │
│         ↑ weights                                │
│                                                  │
│ PyTorch 构建计算图，记录所有操作                  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│          反向传播 (Backward Pass)                │
├─────────────────────────────────────────────────┤
│ loss.backward()                                 │
│                                                  │
│ loss ← cross_entropy ← logits ← embedding       │
│  ↓         ↓            ↓         ↓             │
│ 1.0    ∂L/∂CE      ∂L/∂logits  ∂L/∂weights     │
│                                   ↓             │
│                          存储在 weights.grad     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│          参数更新 (Parameter Update)             │
├─────────────────────────────────────────────────┤
│ optimizer.step()                                │
│                                                  │
│ weights_new = weights_old - lr * weights.grad   │
└─────────────────────────────────────────────────┘
```

### 关键要点总结

| **步骤**                       | **作用**               | **影响的内容**                  |
| ------------------------------ | ---------------------- | ------------------------------- |
| `logits, loss = model(xb, yb)` | 前向传播，计算损失     | 构建计算图                      |
| `optimizer.zero_grad()`        | 清空旧梯度             | `param.grad = None`             |
| `loss.backward()` ⭐           | **计算所有参数的梯度** | **`param.grad = ∂L/∂param`**    |
| `optimizer.step()`             | 用梯度更新参数         | `param.data -= lr * param.grad` |

### 为什么这三步缺一不可？

```python
# 只有 forward，没有 backward
logits, loss = model(xb, yb)
# ❌ 梯度永远是 None，参数不会更新

# 有 backward，没有 step
loss.backward()
# ❌ 只计算了梯度，但参数没有改变

# 有 step，没有 zero_grad
optimizer.step()
# ❌ 梯度会累加，导致错误的更新方向

# ✅ 完整流程
optimizer.zero_grad()  # 清空
loss.backward()         # 计算梯度
optimizer.step()        # 更新参数
```

### 与 model.parameters() 的联系

还记得之前讨论的 `model.parameters()` 吗？

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

- `model.parameters()` 告诉优化器**哪些参数需要更新**
- `loss.backward()` 计算**这些参数的梯度**
- `optimizer.step()` **更新这些参数**

**三者配合才能完成训练！**

### 核心总结

`loss.backward()` 是连接"计算损失"和"更新参数"的关键桥梁：

- 没有它，模型不知道该往哪个方向调整参数
- 它通过链式法则自动计算所有参数的梯度
- 这些梯度被优化器用来更新参数，从而降低损失

---

## 参考资料

- [Attention is All You Need paper](https://arxiv.org/abs/1706.03762)
- [OpenAI GPT-3 paper](https://arxiv.org/abs/2005.14165)
- [OpenAI ChatGPT blog](https://openai.com/blog/chatgpt/)
- [nanoGPT GitHub](https://github.com/karpathy/nanoGPT)

---

**最后更新**: 2025-11-17
