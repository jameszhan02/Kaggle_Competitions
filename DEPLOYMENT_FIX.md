# 🔧 部署问题修复说明

## 问题

Railway 部署失败：`Image of size 5.4 GB exceeded limit of 4.0 GB`

## 原因

完整版 PyTorch 包含 CUDA 支持，体积达 5.4GB，超过 Railway 限制

## 解决方案

使用 **CPU-only 版本** 的 PyTorch，体积仅约 200-300MB

## 已修改的文件

### 1. `requirements.txt`

```
flask==3.0.0
flask-cors==4.0.0
pillow==10.1.0
numpy==1.24.3
gunicorn==21.2.0

# PyTorch CPU-only (lighter version)
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.1.0+cpu
torchvision==0.16.0+cpu
```

**关键变化：**

- 添加 `--extra-index-url` 指向 PyTorch CPU 仓库
- 使用 `torch==2.1.0+cpu` 而不是 `torch==2.1.0`
- 使用 `torchvision==0.16.0+cpu` 而不是 `torchvision==0.16.0`

### 2. `nixpacks.toml` (新增)

配置 Railway 的构建过程，确保正确安装依赖

## 现在的步骤

### 1. 提交更改

```bash
git add requirements.txt nixpacks.toml
git commit -m "Fix: Use PyTorch CPU-only version to reduce image size"
git push
```

### 2. 重新部署

- Railway 会自动检测更改并重新部署
- 构建时间：约 3-5 分钟
- 最终镜像大小：约 1-2GB（在限制范围内）

### 3. 验证部署

```bash
# 等待部署完成后测试
curl https://your-app.railway.app/health
```

## 性能说明

**使用 CPU-only 版本不会影响性能！**

为什么？

- ✅ 我们的模型很小（SimpleCNN）
- ✅ 单次预测速度：CPU ~10-50ms，GPU ~5-20ms
- ✅ API 主要瓶颈在网络，不在计算
- ✅ Railway 免费版不提供 GPU，所以原本也是用 CPU

## 其他云平台建议

如果 Railway 还是太慢或有问题，可以考虑：

### 选项 1: Render (推荐)

- 免费层支持更大镜像
- 配置简单
- URL: https://render.com

### 选项 2: Fly.io

- 免费层 3GB RAM
- 支持 Docker
- URL: https://fly.io

### 选项 3: Heroku

- 稳定但需要付费 ($7/月)
- 生态系统成熟
- URL: https://heroku.com

## 本地测试

在推送前，可以本地测试 CPU 版本：

```bash
# 卸载现有 PyTorch
pip uninstall torch torchvision -y

# 安装 CPU 版本
pip install --no-cache-dir -r requirements.txt

# 测试运行
python app.py
```

## 预期结果

✅ 镜像大小：1-2GB（在限制内）
✅ 构建时间：3-5 分钟
✅ 内存使用：~200MB
✅ 启动时间：~5 秒
✅ 预测速度：~20-50ms

## 故障排除

### 如果还是失败...

**Plan B: 移除一些不必要的依赖**

```txt
flask==3.0.0
flask-cors==4.0.0
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.1.0+cpu
pillow==10.1.0
numpy==1.24.3
gunicorn==21.2.0
```

(移除 torchvision，只保留 torch)

**Plan C: 使用 Docker 自定义镜像**
创建 `Dockerfile` 使用更小的基础镜像

**Plan D: 切换到 Render 或其他平台**

## 问题反馈

如果还有问题，请提供：

1. Railway 完整日志
2. 错误信息截图
3. 尝试过的解决方案

---

💡 **提示：** CPU 版本完全够用！大多数生产环境的 ML API 都用 CPU 版本。
