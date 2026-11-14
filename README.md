# FVSAIT (re-implementation)

按论文《FVSAIT: an Improved Image Translation Model using FHRR-based Encoding
and Lookahead Optimization》(PLOS 投稿) 重新实现的 FVSAIT 代码,以 Meta 的
VSAIT (ECCV 2022) 官方实现为基底。FVSAIT 与 VSAIT 的差异有两点:
**FHRR(傅里叶全息约简表示)复数相位超向量编码** 和 **Lookahead 优化器**。

## 论文 ↔ 代码对照

| 论文章节/公式 | 代码位置 |
|---|---|
| 相位编码 v=e^{iπ(x+1)} (Eq.1) | `models/fhrr.py::phase_encode` |
| 绑定/解绑: 复数逐元素乘/共轭乘 (Eq.2) | `models/fhrr.py::bind / unbind` |
| 叠加: 求和+相位归一化 ph(·) (Eq.3) | `models/fhrr.py::bundle` |
| 相位余弦相似度 sim(a,b)=mean Re(a·b*) (Eq.4) | `models/fhrr.py::similarity` |
| 相似度检索解叠加 (Eq.5, 评估期用) | `models/fhrr.py::retrieve` |
| 编码器 E = VGG19 + MAP-B 投影 + FHRR 相位化 | `models/iit.py::FVSAIT.encode / encode_phases`(MAP-B 即原 LSH 投影,输出 [-1,1] 即相位) |
| 可逆映射网络 R: 两层 1x1 卷积 4096→4096 + ReLU/Tanh,作用于相位表示 | `models/iit.py::FVSAIT.g_net_map / map_phases`(输出 Tanh∈[-1,1] 相位化后绑定;末层初始化 0/1 → U=1 恒等) |
| 判别器: 超向量(相位)上的铰链对抗 (Eq.6-7) | `models/discriminators.py::FVSADiscriminator`(1x1 卷积 [1024,1024,1],输入为相位图) + `losses/da_loss.py` 的 `loss_hinge_gan` |
| 循环一致性 1−sim(v_X, v_G(X)⊗U) (Eq.8) | `losses/da_loss.py::FVSALoss.loss_fhrr_cyc`(复数域相位相似度) |
| 总损失 λ_GAN=1, λ_CYC=10 (Eq.9) | `configs/fvsait.yaml`(`ADV_LOSS_COEF` 默认 1,`VSA_LOSS_COEF: 10.0`) |
| Lookahead: φ=φ+γ(θ−φ), k 步一次慢更新 (Eq.10-11) | `utils/lookahead.py::Lookahead`(标准实现),solver `configure_optimizers` 里按 `cfg.LOOKAHEAD` 包装 Adam |
| 训练超参: Adam(0.5,0.999), lr=4e-4(G/R/D 同), k=5, γ=0.5 | `configs/fvsait.yaml` 的 OPTIMIZERS/LOOKAHEAD |
| 生成器: CUT 骨干(与 VSAIT 相同的 ResNet 生成器) | `models/generators.py::ResNetGenerator`(未加任何注意力) |

## 实现说明(论文未明确处的决策,均可在配置中调整)

1. **相位表示的选择**: MAP-B(LSH)投影输出的余弦相似度天然在 [-1,1],
   与相位 θ=π(x+1) 一一等价。R 和判别器的输入直接用该实值相位表示;
   复数域只用于绑定/解绑与相似度计算(信息无损,梯度可导)。
2. **对抗项输入**: 按论文 Eq.(6)-(7),判别器只看 v_G(X)(相位)与 v_Y(相位),
   不含 VSAIT 原版的 v_X⊙u_XY 附加项。
3. **调度器**: 保留 VSAIT 的 StepLR(每 100k 步减半);论文未提衰减,如不需要
   可将 SCHEDULER.NAME 置空。
4. **消融配置的优化器口径**: MAP-B+Adam / MAP-B+Lookahead / FHRR+Adam 三行
   使用 VSAIT 原优化器设置(1e-4/4e-4, betas 0.0/0.9),只切换各自因子,
   便于因子隔离;完整 FVSAIT 按论文用 4e-4 + (0.5,0.999) + Lookahead。

## 训练 / 测试

```powershell
$vsaitEnv = "C:\Users\chq\AppData\Local\conda\conda\envs\VSAIT"
$env:PATH = "$vsaitEnv;$vsaitEnv\Library\bin;$vsaitEnv\Scripts;" + $env:PATH
cd C:\Users\chq\Desktop\DaVSAIT\FVSAIT_reimpl

# task1: GTA -> Cityscapes (论文: 20 epochs, bs=1, ~100k iters)
& "$vsaitEnv\python.exe" train.py `
  --name fvsait_task1 --data_config ./configs/task1_gta2city_data.yaml `
  --model_config ./configs/fvsait.yaml `
  --aug_config ./configs/gta2cityscapes_aug.yaml `
  --output_dir ./checkpoints --batch_size 1 --seed 42 --max_epochs 20 `
  --val_check_interval 2000

# task2: Aerial Photo <-> Google Maps (论文: 400 epochs, bs=4, 219200 iters)
& "$vsaitEnv\python.exe" train.py `
  --name fvsait_task2_p2m --data_config ./configs/task2_photo2map_data.yaml `
  --model_config ./configs/fvsait.yaml `
  --aug_config ./configs/square256_aug.yaml `
  --output_dir ./checkpoints --batch_size 4 --seed 42 --max_epochs 400 `
  --val_check_interval 2000

# 测试
& "$vsaitEnv\python.exe" test.py --name fvsait_task1_test `
  --checkpoint ./checkpoints/fvsait_task1/version_0/checkpoints/last.ckpt `
  --data_config ./configs/task1_gta2city_data.yaml `
  --model_config ./configs/fvsait.yaml `
  --aug_config ./configs/gta2cityscapes_aug.yaml `
  --batch_size 1 --num_gpus 1 --output_dir ./results
```

## 消融实验配置(论文 Table: 2x2)

| 配置 | 编码 | 优化器 |
|---|---|---|
| `configs/fvsait_mapb_adam.yaml` | MAP-B(=VSAIT) | Adam |
| `configs/fvsait_mapb_lookahead.yaml` | MAP-B | Adam+Lookahead |
| `configs/fvsait_fhrr_adam.yaml` | FHRR | Adam |
| `configs/fvsait.yaml` | FHRR | Adam+Lookahead(完整 FVSAIT) |

## 环境依赖

与 VSAIT 相同(见 requirements.txt),conda 环境 `VSAIT`
(python 3.8 / torch 1.10 / pytorch-lightning 1.5)。

## 致谢与许可

本代码基于 Meta Research 的 [VSAIT](https://github.com/facebookresearch/vsait)
(Theiss et al., ECCV 2022) 实现,遵循其 CC-BY-NC 4.0 许可(见 LICENSE),
仅限非商业使用;VSAIT 及本仓库的修改部分需保留原署名。
