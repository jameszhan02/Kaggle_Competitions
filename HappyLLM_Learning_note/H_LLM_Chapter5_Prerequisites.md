# 第五章前置知识补充：从 Transformer 到 GPT

> **目标读者**：完成第二章 Transformer 学习，准备进入第五章"动手搭建大模型"的 Beginner  
> **参考资料**：[Datawhale Happy-LLM](https://datawhalechina.github.io/happy-llm/)  
> **作者**：AI Assistant  
> **更新日期**：2025-11-05

---

## 📋 目录

1. [学习路线图](#学习路线图)
2. [核心概念 1：Decoder-only 架构](#核心概念-1decoder-only-架构)
3. [核心概念 2：KV Cache（重点）](#核心概念-2kv-cache重点)
4. [核心概念 3：RoPE 位置编码](#核心概念-3rope-位置编码)
5. [核心概念 4：SwiGLU 激活函数](#核心概念-4swiglu-激活函数)
6. [核心概念 5：RMS Norm](#核心概念-5rms-norm)
7. [完整代码示例](#完整代码示例)
8. [常见问题 FAQ](#常见问题-faq)

---

## 学习路线图

```
你已经学过的（第二章）          第五章会遇到的              本文档帮你补充的
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  Transformer    │         │      GPT        │         │  Decoder-only   │
│  (完整架构)      │   →     │  (Decoder-only) │   ←─    │  架构详解       │
│                 │         │                 │         │                 │
│  Multi-Head     │         │  Grouped Query  │         │  KV Cache       │
│  Attention      │   →     │  Attention      │   ←─    │  机制详解       │
│                 │         │                 │         │                 │
│  Position       │         │      RoPE       │         │  RoPE 简化      │
│  Encoding       │   →     │  (旋转编码)      │   ←─    │  讲解           │
│                 │         │                 │         │                 │
│  ReLU + FFN     │         │     SwiGLU      │         │  激活函数       │
│                 │   →     │                 │   ←─    │  演化史         │
│                 │         │                 │         │                 │
│  Layer Norm     │         │    RMS Norm     │         │  归一化方法     │
│                 │   →     │                 │   ←─    │  对比           │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

---

## 核心概念 1：Decoder-only 架构

### 🎯 为什么要 Decoder-only？

回忆一下你在第二章学过的完整 Transformer：

```
完整 Transformer = Encoder + Decoder

Encoder（编码器）:
- 输入：源语言句子 "I love you"
- 输出：理解后的表示向量
- 特点：双向注意力（可以看前后所有词）

Decoder（解码器）:
- 输入：目标语言开头 "我"
- 输出：预测下一个词 "爱"
- 特点：单向注意力（只能看前面的词，不能偷看后面）
```

**GPT 的选择**：只用 Decoder！

```
GPT = Decoder + Decoder + Decoder + ... (只堆叠 Decoder 层)

为什么？
✅ 任务是"文本生成"（预测下一个词）
✅ 生成时不需要"理解"另一种语言（不需要 Encoder）
✅ 只需要根据前文预测后文（Decoder 就够了）
```

### 📊 三种架构对比

| 特性           | Encoder-only<br>(BERT) | Decoder-only<br>(GPT) | Encoder-Decoder<br>(T5)      |
| -------------- | ---------------------- | --------------------- | ---------------------------- |
| **注意力方式** | 双向（看所有词）       | 单向（只看前面）      | Encoder 双向 + Decoder 单向  |
| **Mask 矩阵**  | ❌ 不需要              | ✅ 需要（上三角）     | Encoder 不需要，Decoder 需要 |
| **擅长任务**   | 文本分类、NER          | 文本生成、对话        | 翻译、摘要                   |
| **代表模型**   | BERT, RoBERTa          | GPT-2/3/4, LLaMA      | T5, BART                     |
| **训练目标**   | MLM（填空）            | CLM（预测下一词）     | Seq2Seq                      |

### 🔍 关键代码对比

**你第二章学过的 `is_causal` 参数**：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal=False):
        #            ↑ 这个参数决定是 Encoder 还是 Decoder！
        super().__init__()
        self.is_causal = is_causal

        if is_causal:
            # Decoder 需要 mask（只能看前面的词）
            mask = torch.full((1, 1, max_len, max_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)  # 上三角
            self.register_buffer("mask", mask)
```

**BERT (Encoder-only)**：`is_causal=False`  
**GPT (Decoder-only)**：`is_causal=True`

**Mask 的作用**（你第二章学过的）：

```
假设输入："今天 天气 很 好"

没有 Mask（BERT）:
今天 → 可以看到：今天, 天气, 很, 好  ✅ 双向
天气 → 可以看到：今天, 天气, 很, 好  ✅ 双向
很   → 可以看到：今天, 天气, 很, 好  ✅ 双向
好   → 可以看到：今天, 天气, 很, 好  ✅ 双向

有 Mask（GPT）:
今天 → 只能看到：今天               ✅ 单向
天气 → 只能看到：今天, 天气         ✅ 单向
很   → 只能看到：今天, 天气, 很     ✅ 单向
好   → 只能看到：今天, 天气, 很, 好 ✅ 单向
```

---

## 核心概念 2：KV Cache（重点）

> ⚠️ **这是第五章最难理解的部分！但理解了它，你就理解了 LLM 生成的核心优化！**

### 🤔 问题：为什么需要 KV Cache？

想象你在用 GPT 生成一句话：

```
输入："今天天气"
期望输出："今天天气很好，适合出去玩。"

生成过程（一次生成一个词）：
Step 1: "今天天气" → 预测 → "很"
Step 2: "今天天气很" → 预测 → "好"
Step 3: "今天天气很好" → 预测 → "，"
Step 4: "今天天气很好，" → 预测 → "适合"
... (每次都要重新计算所有前面的词！)
```

### ❌ 没有 KV Cache 的问题

每一步都要**重新计算所有前面的词**的 Key 和 Value：

```python
# Step 1: 输入 "今天天气"
input_1 = ["今天", "天气"]
K_1 = compute_key(input_1)      # 计算 ["今天"的K, "天气"的K]
V_1 = compute_value(input_1)    # 计算 ["今天"的V, "天气"的V]
Q_1 = compute_query("天气")     # 只需要最后一个词的 Q
output_1 = attention(Q_1, K_1, V_1)  # 预测 "很"

# Step 2: 输入 "今天天气很"
input_2 = ["今天", "天气", "很"]
K_2 = compute_key(input_2)      # 又计算了一遍 "今天" 和 "天气" 的 K！❌
V_2 = compute_value(input_2)    # 又计算了一遍 "今天" 和 "天气" 的 V！❌
Q_2 = compute_query("很")
output_2 = attention(Q_2, K_2, V_2)  # 预测 "好"

# Step 3: 输入 "今天天气很好"
input_3 = ["今天", "天气", "很", "好"]
K_3 = compute_key(input_3)      # 又又又计算了一遍前面所有词！❌❌❌
V_3 = compute_value(input_3)    # 浪费计算！
...
```

**问题**：

- 每次都重复计算前面的 K 和 V
- 生成 100 个词，前面的词会被重复计算 99 次！
- 浪费大量计算资源和时间

### ✅ 有 KV Cache 的优化

**核心思想**：把已经计算过的 K 和 V **缓存起来**，下次直接用！

```python
# 初始化空缓存
cache_K = []
cache_V = []

# Step 1: 输入 "今天天气"
new_K_1 = compute_key(["今天", "天气"])
new_V_1 = compute_value(["今天", "天气"])
cache_K = new_K_1  # 保存到缓存
cache_V = new_V_1  # 保存到缓存
Q_1 = compute_query("天气")
output_1 = attention(Q_1, cache_K, cache_V)  # 预测 "很"

# Step 2: 输入新词 "很"
new_K_2 = compute_key(["很"])         # 只计算新词！✅
new_V_2 = compute_value(["很"])       # 只计算新词！✅
cache_K = concat(cache_K, new_K_2)    # 拼接到缓存
cache_V = concat(cache_V, new_V_2)    # 拼接到缓存
Q_2 = compute_query("很")
output_2 = attention(Q_2, cache_K, cache_V)  # 预测 "好"

# Step 3: 输入新词 "好"
new_K_3 = compute_key(["好"])         # 只计算新词！✅
new_V_3 = compute_value(["好"])       # 只计算新词！✅
cache_K = concat(cache_K, new_K_3)    # 拼接到缓存
cache_V = concat(cache_V, new_V_3)    # 拼接到缓存
...
```

**优化效果**：

- ✅ 每个词的 K 和 V **只计算一次**
- ✅ 生成速度提升 **数十倍**！
- ✅ 这就是为什么 ChatGPT 能快速生成的秘密

### 📊 性能对比

```
生成 100 个词的计算量：

没有 KV Cache:
计算次数 = 1 + 2 + 3 + ... + 100 = 5050 次 K/V 计算 ❌

有 KV Cache:
计算次数 = 100 次 K/V 计算 ✅

加速比 = 5050 / 100 = 50.5 倍！🚀
```

### 💻 代码实现

```python
class MultiHeadAttentionWithCache(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cache=None):
        """
        参数:
            x: 输入，shape = (batch, seq_len, dim)
            cache: 缓存的 K 和 V，格式 {'k': tensor, 'v': tensor}

        返回:
            output: 输出
            new_cache: 更新后的缓存
        """
        batch_size, seq_len, _ = x.shape

        # 计算新的 Q, K, V
        q = self.wq(x)  # (batch, seq_len, dim)
        k_new = self.wk(x)  # (batch, seq_len, dim)
        v_new = self.wv(x)  # (batch, seq_len, dim)

        # 如果有缓存，拼接历史的 K 和 V
        if cache is not None:
            k = torch.cat([cache['k'], k_new], dim=1)  # 拼接历史
            v = torch.cat([cache['v'], v_new], dim=1)
        else:
            k = k_new
            v = v_new

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention 计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.wo(out)

        # 保存当前的 K 和 V 到缓存（供下次使用）
        new_cache = {'k': k.transpose(1, 2), 'v': v.transpose(1, 2)}

        return out, new_cache


# 使用示例
model = MultiHeadAttentionWithCache(dim=512, n_heads=8)
cache = None  # 初始化空缓存

# 模拟生成过程
input_ids = tokenizer.encode("今天天气")

for step in range(10):  # 生成 10 个词
    # 只输入新词（第一次输入所有词）
    if step == 0:
        x = embed(input_ids)  # (1, 2, 512)
    else:
        x = embed([new_token_id])  # (1, 1, 512) 只输入新词！

    # 前向传播（自动使用缓存）
    output, cache = model(x, cache=cache)  # cache 会自动累积

    # 预测下一个词
    new_token_id = output.argmax(dim=-1)
    input_ids.append(new_token_id)
```

### 🎨 图解 KV Cache

```
时刻 t=1: 输入 "今天"
┌─────────────────────────────────────┐
│ Input: "今天"                        │
│ Compute: Q1, K1, V1                 │
│ Cache: K1, V1                       │ ← 保存到缓存
│ Attention(Q1, K1, V1) → predict     │
└─────────────────────────────────────┘

时刻 t=2: 输入 "天气" (新词)
┌─────────────────────────────────────┐
│ Input: "天气"                        │
│ Compute: Q2, K2, V2 (只计算新词!)    │
│ Cache: [K1, K2], [V1, V2]           │ ← 拼接到缓存
│ Attention(Q2, [K1,K2], [V1,V2])     │ ← 用完整缓存
└─────────────────────────────────────┘

时刻 t=3: 输入 "很" (新词)
┌─────────────────────────────────────┐
│ Input: "很"                          │
│ Compute: Q3, K3, V3 (只计算新词!)    │
│ Cache: [K1,K2,K3], [V1,V2,V3]       │ ← 继续拼接
│ Attention(Q3, [K1,K2,K3], [V1,V2,V3])│
└─────────────────────────────────────┘
```

---

## 核心概念 3：RoPE 位置编码

### 🎯 为什么需要位置编码？

回忆一下：Self-Attention 本身**没有位置信息**！

```
句子 A: "我 爱 你"
句子 B: "你 爱 我"

如果没有位置编码，Self-Attention 会认为它们一样！
因为包含的词相同，只是顺序不同。
```

### 📝 你第二章学过的位置编码（Sinusoidal）

```python
# 原始 Transformer 的位置编码（加法）
def get_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                        -(math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# 使用方式：直接加到 embedding 上
x = token_embedding + positional_encoding
```

**问题**：

- 位置信息和内容信息"混在一起"了
- 难以泛化到更长的序列

### 🔄 RoPE (Rotary Position Embedding)

**核心思想**：通过**旋转向量**来编码位置信息

```
传统方式：x + pos_encoding        （加法）
RoPE方式：rotate(x, θ)            （旋转）

其中旋转角度 θ 取决于位置：
位置 0 → 旋转 0°
位置 1 → 旋转 θ
位置 2 → 旋转 2θ
位置 3 → 旋转 3θ
...
```

### 🎨 直观理解

想象每个词向量是二维平面上的箭头：

```
位置编码 = 旋转箭头

原始向量 "今天" at 位置 0:
  →  (不旋转)

"今天" at 位置 1:
  ↗  (旋转 30°)

"今天" at 位置 2:
  ↑  (旋转 60°)

"今天" at 位置 3:
  ↖  (旋转 90°)
```

**关键性质**：两个向量的**相对位置**可以通过**相对旋转角度**表示！

```
词 A 在位置 1 (旋转 30°)
词 B 在位置 3 (旋转 90°)
相对角度 = 90° - 30° = 60°

无论 A 和 B 在哪个位置，只要相对距离是 2，
相对角度总是 60°！

这让模型更容易学习相对位置关系！
```

### 💻 RoPE 简化实现

```python
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    预计算旋转频率

    参数:
        dim: 向量维度（通常是 head_dim）
        max_seq_len: 最大序列长度
        theta: 基础频率（越大，旋转越慢）
    """
    # 计算每个维度的频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # freqs shape: (dim/2,)

    # 计算每个位置的角度
    t = torch.arange(max_seq_len)  # [0, 1, 2, ..., max_seq_len-1]
    # t shape: (max_seq_len,)

    # 外积：位置 × 频率 = 角度
    freqs = torch.outer(t, freqs).float()
    # freqs shape: (max_seq_len, dim/2)

    # 转换为复数形式（用于旋转）
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # freqs_cis shape: (max_seq_len, dim/2)

    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    """
    应用旋转位置编码

    参数:
        x: 输入向量，shape = (..., seq_len, dim)
        freqs_cis: 预计算的旋转频率

    返回:
        旋转后的向量
    """
    # 将实数向量转换为复数（每两个维度一组）
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )
    # x_complex shape: (..., seq_len, dim/2)

    # 应用旋转（复数乘法 = 旋转）
    x_rotated = x_complex * freqs_cis

    # 转换回实数
    x_out = torch.view_as_real(x_rotated).flatten(-2)

    return x_out.type_as(x)


# 在 Attention 中使用
class AttentionWithRoPE(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.head_dim = dim // n_heads

        # 预计算旋转频率
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)

    def forward(self, q, k, v, start_pos=0):
        # 获取当前序列的旋转频率
        seq_len = q.size(1)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]

        # 只对 Q 和 K 应用 RoPE（V 不需要）
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # 正常的 Attention 计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        return out
```

### 📊 RoPE vs 传统位置编码

| 特性         | 传统 Sinusoidal  | RoPE                   |
| ------------ | ---------------- | ---------------------- |
| **编码方式** | 加法 (x + PE)    | 旋转 (rotate)          |
| **位置信息** | 绝对位置         | 相对位置               |
| **泛化能力** | 较差             | 更好                   |
| **长度外推** | 困难             | 容易                   |
| **使用模型** | 原始 Transformer | GPT-NeoX, LLaMA, GPT-J |

---

## 核心概念 4：SwiGLU 激活函数

### 📝 你第二章学过的 FFN（前馈神经网络）

```python
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # 简单的两层结构
        return self.w2(F.relu(self.w1(x)))
        #               ↑ ReLU 激活函数
```

**问题**：

- ReLU 在负数区域梯度为 0（死亡 ReLU 问题）
- 可能限制模型的表达能力

### 🔄 激活函数的演化

```
1. ReLU (2012)
   f(x) = max(0, x)

   优点：简单，计算快
   缺点：负数梯度为 0

2. GELU (2016) [BERT 使用]
   f(x) = x · Φ(x)  (Φ 是标准正态分布的CDF)

   优点：更平滑，性能更好
   缺点：计算稍慢

3. Swish / SiLU (2017) [接近 GELU]
   f(x) = x · sigmoid(x)

   优点：简单，性能好

4. GLU (Gated Linear Unit, 2017)
   f(x) = x ⊙ sigmoid(Wx)

   优点：引入门控机制
   缺点：需要额外的参数矩阵

5. SwiGLU (2020) [LLaMA, PaLM 使用]
   f(x) = Swish(W1·x) ⊙ (W3·x)

   优点：结合 Swish 和 GLU 的优点
```

### 💻 SwiGLU 实现

```python
class SwiGLU_FFN(nn.Module):
    """
    SwiGLU = Swish(W1·x) ⊙ W3·x

    相比传统 FFN，多了一个线性层 W3
    """
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        # 三个线性层（传统 FFN 只有两个）
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)      # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)      # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)      # Up projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU 计算
        # 1. 通过 W1 计算门控信号
        gate = F.silu(self.w1(x))  # Swish/SiLU 激活

        # 2. 通过 W3 计算特征
        features = self.w3(x)

        # 3. 门控：逐元素相乘
        hidden = gate * features

        # 4. 投影回原始维度
        output = self.w2(hidden)
        output = self.dropout(output)

        return output


# 对比：传统 FFN
class Traditional_FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = F.relu(self.w1(x))  # ReLU 激活
        output = self.w2(hidden)
        output = self.dropout(output)
        return output
```

### 🎨 可视化对比

```
传统 FFN (ReLU):
Input (dim)
    ↓
  W1 (Linear)
    ↓
  ReLU
    ↓
  W2 (Linear)
    ↓
Output (dim)

参数量: W1 + W2


SwiGLU FFN:
Input (dim)
    ↓
    ├──→ W1 → Swish ──┐
    │                  ↓
    └──→ W3 ──────→  ⊙  (逐元素相乘)
                      ↓
                     W2
                      ↓
                  Output (dim)

参数量: W1 + W2 + W3  (多了 33%，但性能提升更多！)
```

### 📊 性能对比（实验结果）

在 LLaMA 论文中的实验：

| 激活函数   | PPL (越低越好) | 参数量 |
| ---------- | -------------- | ------ |
| ReLU       | 9.8            | 1.0x   |
| GELU       | 9.5            | 1.0x   |
| **SwiGLU** | **9.2** ✅     | 1.33x  |

**结论**：多 33% 参数，但性能提升显著，性价比高！

---

## 核心概念 5：RMS Norm

### 📝 你第二章学过的 Layer Norm

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 计算均值和标准差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # 标准化
        x_norm = (x - mean) / (std + self.eps)

        # 缩放和平移
        return self.a * x_norm + self.b
```

**Layer Norm 做了什么？**

1. **中心化**：减去均值 (x - mean)
2. **标准化**：除以标准差 (/ std)
3. **缩放和平移**：可学习参数 a 和 b

### 🔄 RMS Norm (Root Mean Square Norm)

**核心思想**：只做标准化，不做中心化！

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 计算 RMS (均方根)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # 标准化（不减均值！）
        x_norm = x / rms

        # 只缩放，不平移
        return self.weight * x_norm
```

### 📊 公式对比

**Layer Norm**:

```
LN(x) = γ · (x - μ) / σ + β

其中:
μ = mean(x)         # 均值
σ = std(x)          # 标准差
γ, β 是可学习参数
```

**RMS Norm**:

```
RMS(x) = γ · x / RMS(x)

其中:
RMS(x) = sqrt(mean(x²))  # 均方根
γ 是可学习参数（没有 β！）
```

### ⚡ 为什么 RMS Norm 更好？

| 特性           | Layer Norm              | RMS Norm             |
| -------------- | ----------------------- | -------------------- |
| **计算复杂度** | 高（需要算 mean + std） | 低（只需要算 RMS）   |
| **参数数量**   | 2 × dim (γ 和 β)        | 1 × dim (只有 γ)     |
| **训练速度**   | 慢                      | 快 5-10% ⚡          |
| **效果**       | 好                      | 几乎一样好           |
| **稳定性**     | 好                      | 更好（不需要中心化） |

**关键洞察**：

- 在 Transformer 中，**中心化不是必需的**！
- 去掉 mean 计算可以加速，且不影响性能
- 大模型训练时，每一点加速都很重要

### 💻 完整对比代码

```python
import torch
import torch.nn as nn
import time

# 1. Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

# 2. RMS Norm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x / rms

# 性能测试
def benchmark():
    dim = 4096
    batch_size = 32
    seq_len = 2048

    x = torch.randn(batch_size, seq_len, dim).cuda()

    ln = LayerNorm(dim).cuda()
    rms = RMSNorm(dim).cuda()

    # 预热
    for _ in range(10):
        _ = ln(x)
        _ = rms(x)

    # Layer Norm 测速
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = ln(x)
    torch.cuda.synchronize()
    ln_time = time.time() - start

    # RMS Norm 测速
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = rms(x)
    torch.cuda.synchronize()
    rms_time = time.time() - start

    print(f"Layer Norm: {ln_time:.4f}s")
    print(f"RMS Norm:   {rms_time:.4f}s")
    print(f"Speedup:    {ln_time/rms_time:.2f}x")

# 运行测试
# benchmark()
# 输出示例:
# Layer Norm: 0.1234s
# RMS Norm:   0.1089s
# Speedup:    1.13x
```

---

## 完整代码示例

### 🎯 构建一个简化版 GPT Block

把上面所有概念整合到一起：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x / rms


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """预计算 RoPE 的旋转频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    """应用 RoPE"""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim/2)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)


class MultiHeadAttentionWithCache(nn.Module):
    """带 KV Cache 和 RoPE 的多头注意力"""
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # 预计算 RoPE 频率
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)

        # 注册 causal mask
        mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x, cache=None, start_pos=0):
        """
        参数:
            x: 输入，shape = (batch, seq_len, dim)
            cache: KV 缓存
            start_pos: 当前位置（用于 RoPE 和 mask）
        """
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len].to(x.device)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # KV Cache
        if cache is not None:
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用 causal mask
        total_len = k.size(2)
        scores = scores + self.mask[:, :, start_pos:start_pos+seq_len, :total_len]

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.wo(out)

        # 更新缓存
        new_cache = {'k': k, 'v': v}

        return out, new_cache


class SwiGLU_FFN(nn.Module):
    """SwiGLU 前馈网络"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """完整的 Transformer Block (GPT 风格)"""
    def __init__(self, dim, n_heads, hidden_dim, max_seq_len):
        super().__init__()

        # Attention 部分
        self.attention = MultiHeadAttentionWithCache(dim, n_heads, max_seq_len)
        self.attention_norm = RMSNorm(dim)

        # FFN 部分
        self.ffn = SwiGLU_FFN(dim, hidden_dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, cache=None, start_pos=0):
        # Attention + Residual
        h, new_cache = self.attention(
            self.attention_norm(x),
            cache=cache,
            start_pos=start_pos
        )
        x = x + h

        # FFN + Residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_cache


# 使用示例
if __name__ == "__main__":
    # 配置
    dim = 512
    n_heads = 8
    hidden_dim = 2048
    max_seq_len = 2048

    # 创建模型
    block = TransformerBlock(dim, n_heads, hidden_dim, max_seq_len)

    # 模拟生成过程
    batch_size = 1
    vocab_size = 50000

    # 假设已有 embedding 层
    embedding = nn.Embedding(vocab_size, dim)

    # 初始输入："今天天气" (假设 token ids = [1234, 5678])
    input_ids = torch.tensor([[1234, 5678]])
    x = embedding(input_ids)  # (1, 2, 512)

    cache = None
    start_pos = 0

    # 第一次前向（处理初始输入）
    print("Step 1: 处理 '今天天气'")
    x, cache = block(x, cache=cache, start_pos=start_pos)
    print(f"Output shape: {x.shape}")
    print(f"Cache K shape: {cache['k'].shape}")
    start_pos += x.size(1)

    # 后续生成（每次只输入一个新 token）
    for step in range(5):
        print(f"\nStep {step+2}: 生成新词")

        # 模拟预测的新 token
        new_token_id = torch.randint(0, vocab_size, (1, 1))
        x = embedding(new_token_id)  # (1, 1, 512) 只输入一个新词！

        # 前向传播（复用缓存）
        x, cache = block(x, cache=cache, start_pos=start_pos)
        print(f"Output shape: {x.shape}")
        print(f"Cache K shape: {cache['k'].shape}")  # 缓存在增长！

        start_pos += 1

    print("\n✅ 完整的生成流程演示完成！")
```

### 📊 输出示例

```
Step 1: 处理 '今天天气'
Output shape: torch.Size([1, 2, 512])
Cache K shape: torch.Size([1, 8, 2, 64])

Step 2: 生成新词
Output shape: torch.Size([1, 1, 512])
Cache K shape: torch.Size([1, 8, 3, 64])  ← 缓存增长了！

Step 3: 生成新词
Output shape: torch.Size([1, 1, 512])
Cache K shape: torch.Size([1, 8, 4, 64])  ← 继续增长

Step 4: 生成新词
Output shape: torch.Size([1, 1, 512])
Cache K shape: torch.Size([1, 8, 5, 64])

Step 5: 生成新词
Output shape: torch.Size([1, 1, 512])
Cache K shape: torch.Size([1, 8, 6, 64])

Step 6: 生成新词
Output shape: torch.Size([1, 1, 512])
Cache K shape: torch.Size([1, 8, 7, 64])

✅ 完整的生成流程演示完成！
```

---

## 常见问题 FAQ

### Q1: 为什么 RoPE 只应用在 Q 和 K 上，不应用在 V 上？

**A**: RoPE 的目的是编码**位置关系**，让模型知道"词 A 和词 B 之间的距离"。

- **Q 和 K** 用于计算注意力权重（谁和谁相关）
  - 需要位置信息，因为相对位置影响相关性
- **V** 是实际的内容值
  - 不需要位置信息，内容本身不因位置改变

类比：

```
Q: "请给我附近的咖啡店"
K: [星巴克(500米), 咖啡厅A(2公里), 咖啡厅B(100米)]
   ↑ 需要知道距离（位置）
V: [星巴克的菜单, 咖啡厅A的菜单, 咖啡厅B的菜单]
   ↑ 菜单内容不随距离改变
```

### Q2: KV Cache 会不会占用太多内存？

**A**: 会的！这是 LLM 推理的主要瓶颈之一。

**内存占用计算**：

```
每个 token 的 KV 缓存大小 = 2 × n_layers × dim × 2 bytes (FP16)

假设 LLaMA-7B:
- n_layers = 32
- dim = 4096
- 生成 2048 个 tokens

KV Cache = 2 × 32 × 4096 × 2048 × 2 bytes
         ≈ 1 GB

生成越长，占用越大！
```

**优化方法**：

- **Grouped Query Attention (GQA)**：减少 K 和 V 的头数
- **Multi-Query Attention (MQA)**：所有头共享一个 K 和 V
- **PagedAttention**：分页管理缓存（vLLM 使用）

### Q3: 为什么大模型都用 RMSNorm 而不是 LayerNorm？

**A**: 主要是**速度**！

在大模型中：

- 训练成本 = 数千万美元
- 加速 5% = 节省数百万美元
- RMSNorm 几乎不影响效果，但能加速 5-10%

性价比极高！

### Q4: SwiGLU 比 ReLU 好多少？

**A**: 根据实验（LLaMA, PaLM 论文）：

- 在小模型（<1B）：提升不明显
- 在大模型（>10B）：提升显著（PPL 降低 2-5%）

原因：大模型需要更强的非线性能力，SwiGLU 的门控机制更有用。

### Q5: 我该按什么顺序学习？

**A**: 推荐顺序：

1. ✅ **复习第二章的 Mask 和 Multi-Head Attention**（1 天）
2. ✅ **重点理解 KV Cache**（2-3 天，画图！）
3. ✅ **了解 RoPE 的作用**（1 天，不需要深究数学）
4. ✅ **快速了解 SwiGLU 和 RMSNorm**（半天）
5. ✅ **运行完整代码示例**（1 天，调试理解）
6. ✅ **回去看第五章**（这时候就看懂了！）

---

## 📚 推荐资源

### 论文

- **RoPE**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **SwiGLU**: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- **LLaMA**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **KV Cache**: 搜索 "KV Cache optimization" 相关论文

### 代码

- **LLaMA 官方实现**: [facebookresearch/llama](https://github.com/facebookresearch/llama)
- **nanoGPT**: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (简化版，适合学习)
- **Transformers 库**: [huggingface/transformers](https://github.com/huggingface/transformers)

### 视频

- **Andrej Karpathy - Let's build GPT**: [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- **3Blue1Brown - Attention in transformers**: [YouTube](https://www.youtube.com/watch?v=eMlx5fFNoYc)

---

## 🎓 总结

### 从 Transformer (第二章) 到 GPT (第五章) 的演化

| 组件         | 第二章学的           | 第五章用的   | 核心改进     |
| ------------ | -------------------- | ------------ | ------------ |
| **架构**     | Encoder + Decoder    | Decoder-only | 专注生成任务 |
| **注意力**   | Multi-Head Attention | + KV Cache   | 生成加速 50x |
| **位置编码** | Sinusoidal (加法)    | RoPE (旋转)  | 更好的泛化   |
| **激活函数** | ReLU                 | SwiGLU       | 表达能力更强 |
| **归一化**   | LayerNorm            | RMSNorm      | 速度快 10%   |

### 最重要的三个概念（按优先级）

1. ⭐⭐⭐ **KV Cache**: 理解生成过程的核心，必须掌握！
2. ⭐⭐ **Decoder-only 架构**: 为什么只用 Decoder
3. ⭐ **RoPE/SwiGLU/RMSNorm**: 知道是改进版就行，细节可以后续深入

### 下一步

现在你已经掌握了所有前置知识，可以：

1. 回去看第五章的代码，应该能看懂了
2. 尝试运行本文档的代码示例
3. 修改参数，观察效果
4. 阅读 LLaMA 或 nanoGPT 的源码

**加油！你已经具备搭建大模型的基础了！** 🚀

---

## 📝 笔记区域

> 在这里记录你的学习心得和疑问：

```
我的笔记:
-


待解决的问题:
-


代码实验记录:
-

```

---

**创建日期**: 2025-11-05  
**最后更新**: 2025-11-05  
**版本**: 1.0  
**反馈**: 如有问题或建议，欢迎提出！
