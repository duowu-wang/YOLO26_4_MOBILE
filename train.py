"""
YOLO26n-seg Baseline

运行：
python train.py yolo26n-seg.pt Cracks_yolo_nas/data.yaml yolo26n_seg_baseline

说明：
标准 YOLO26n-seg + 官方预训练权重 + 裂缝数据集。

作为论文 Baseline 使用时，建议后续消融实验保持以下条件一致：
1. 数据集划分不变；
2. 输入尺寸 imgsz 不变；
3. batch 不变；
4. epochs 不变；
5. optimizer 不变；
6. seed 不变；
7. 数据增强策略不变。

后续仅修改网络结构或某个指定模块，保证消融实验具有可比性。
"""

import warnings
import os
import sys

from ultralytics import YOLO

# 忽略非关键警告信息，使控制台输出更加简洁
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    # sys.argv 用于读取命令行传入的参数
    # 例如：
    # python train.py yolo26n-seg.pt Cracks_yolo_nas/data.yaml yolo26n_seg_baseline
    #
    # args[0] = train.py
    # args[1] = yolo26n-seg.pt
    # args[2] = Cracks_yolo_nas/data.yaml
    # args[3] = yolo26n_seg_baseline
    args = sys.argv

    # ================= 路径配置 =================

    # 官方 YOLO26n-seg 预训练权重路径
    # 如果命令行传入权重名称，则从 ./weights/ 目录中读取
    # 如果未传入，则默认使用 ./weights/yolo26n-seg.pt
    pretrained_weights_path = "./weights/" + args[1] if len(args) > 1 else "./weights/yolo26n-seg.pt"

    # 数据集 YAML 配置文件路径
    # 例如 data.yaml 中通常包含：
    # train: 训练集路径
    # val:   验证集路径
    # nc:    类别数量
    # names: 类别名称
    data_yaml_path = "./datasets/" + args[2] if len(args) > 2 else "./datasets/Cracks_yolo_nas/data.yaml"

    # 当前训练实验名称
    # 最终结果会保存在：
    # runs/train/yolo26n_seg_baseline/
    run_name = args[3] if len(args) > 3 else "yolo26n_seg_baseline"

    # ================= 文件检查 =================

    # 检查预训练权重是否存在，避免路径错误导致后续模型加载失败
    if not os.path.exists(pretrained_weights_path):
        raise FileNotFoundError(f"预训练权重不存在: {pretrained_weights_path}")

    # 检查数据集配置文件是否存在
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"数据集 YAML 不存在: {data_yaml_path}")

    # ================= 模型初始化 =================

    print("🚀 加载 YOLO26n-seg 官方预训练模型...")

    # 直接加载官方 yolo26n-seg.pt
    # 模型结构和预训练参数都会从 .pt 文件中自动恢复
    # 对于 Baseline，这是最直接、最规范的加载方式
    model = YOLO(pretrained_weights_path)

    print("✅ 模型加载完成，开始训练")

    # ================= 开始训练 =================

    results = model.train(
        # -------------------- 数据集 --------------------

        # 数据集 YAML 文件路径
        # 用于告诉 Ultralytics 训练集、验证集、类别数量以及类别名称
        data=data_yaml_path,

        # -------------------- 输入图像 --------------------

        # 网络输入图像尺寸
        # imgsz=640 表示训练时图像会被缩放/填充至 640×640
        # 640 是 YOLO 系列较常用的标准输入尺寸
        # 对裂缝这种细长目标，如果未来研究小裂缝识别，也可以单独对比 800、960 等尺寸
        # 但 Baseline 一旦确定，后续消融实验应保持一致
        imgsz=640,

        # -------------------- 训练轮数 --------------------

        # 最大训练 Epoch 数
        # 一个 Epoch 表示完整遍历一次训练集
        # 150 对迁移学习来说通常已经能够充分收敛
        epochs=150,

        # -------------------- Batch Size --------------------

        # 每次送入 GPU 进行一次前向传播和反向传播的图像数量
        # batch 越大：
        # 1. GPU 显存占用越高
        # 2. 单位时间吞吐量通常越高
        # 3. 梯度估计更加稳定
        #
        # 你的 RTX 5060 Ti 具有 16 GB 显存
        # 对 YOLO26n-seg + 640×640 输入，建议优先使用 batch=32
        #
        # 如果后续改进网络后出现 CUDA Out Of Memory，可以统一降至：
        # batch=16
        #
        # 为保证论文消融实验公平，最终最好所有模型统一 batch
        batch=32,

        # -------------------- GPU --------------------

        # 指定训练设备
        # device=0 表示使用第 1 张 NVIDIA GPU
        #
        # 单显卡机器通常直接使用 0
        # 如果存在多张 GPU，例如可使用：
        # device=[0, 1]
        device=0,

        # -------------------- DataLoader --------------------

        # 数据加载进程数
        # workers 越多，CPU 可以并行读取和预处理更多图像
        #
        # 但并不是越大越好，尤其 Windows 下多进程 DataLoader
        # workers 过大可能导致：
        # 1. 内存占用增加
        # 2. 训练启动变慢
        # 3. 偶发卡死
        #
        # 你的 Core Ultra 5 245K 性能足够，workers=8 较为稳妥
        workers=8,

        # -------------------- 数据缓存 --------------------

        # 将预处理后的数据缓存到磁盘，提高后续 Epoch 的数据读取速度
        #
        # cache="disk"：
        # 将缓存保存到磁盘，不大量占用系统 RAM
        #
        # 你的电脑只有 32 GB RAM，同时 Windows、Python、IDE
        # 以及 DataLoader 本身也会消耗内存，因此 disk 模式更加稳妥
        #
        # 其他可选方式：
        # cache=False  -> 不缓存
        # cache="ram"  -> 缓存到内存，速度最快，但 RAM 占用较高
        cache="disk",

        # -------------------- 混合精度训练 --------------------

        # AMP = Automatic Mixed Precision
        # 自动混合精度训练
        #
        # 训练过程中同时使用 FP16 和 FP32
        # 主要作用：
        # 1. 降低显存占用
        # 2. 提高 GPU 计算吞吐率
        # 3. 通常不会明显降低模型精度
        #
        # NVIDIA GPU 训练 YOLO 时一般建议开启
        amp=True,

        # -------------------- 优化器 --------------------

        # 自动选择优化器及相关参数
        #
        # optimizer="auto" 会由 Ultralytics 根据训练配置
        # 自动选择较合适的优化器和学习率策略
        #
        # Baseline 阶段不建议同时手动修改：
        # lr0、momentum、weight_decay 等参数
        #
        # 原因是这样可以减少人为超参数干预，
        # 后续网络结构改进的实验结果更容易解释
        optimizer="auto",

        # -------------------- Early Stopping --------------------

        # 早停耐心值
        #
        # patience=50 表示：
        # 如果模型连续 50 个 Epoch 的验证指标没有得到改善，
        # 则提前终止训练
        #
        # 好处：
        # 1. 避免模型已经收敛后继续浪费训练时间
        # 2. 降低后期过拟合风险
        #
        # 如果始终有提升，则最多训练到 epochs=150
        patience=50,

        # -------------------- Mosaic --------------------

        # 在最后若干个 Epoch 关闭 Mosaic 数据增强
        #
        # Mosaic 会将多张图像拼接为一张训练图像，
        # 有利于提高目标尺度和场景多样性
        #
        # 但训练后期继续使用较强 Mosaic 可能影响模型对
        # 真实图像边界和 Mask 轮廓的精细学习
        #
        # close_mosaic=10 表示：
        # 最后 10 个 Epoch 关闭 Mosaic，让模型在真实图像分布上微调
        close_mosaic=10,

        # -------------------- 随机种子 --------------------

        # 固定随机种子
        #
        # 会影响：
        # 数据增强随机过程
        # 数据顺序
        # 权重初始化中的随机行为等
        #
        # 固定 seed 可以提高实验可复现性
        # 对论文消融实验尤其重要
        seed=0,

        # -------------------- 确定性计算 --------------------

        # 启用更加确定性的训练行为
        #
        # deterministic=True 有利于：
        # 相同环境 + 相同参数 + 相同随机种子时
        # 尽可能获得一致结果
        #
        # 缺点是某些情况下会略微降低训练速度
        # 但论文实验通常更建议保证可复现性
        deterministic=True,

        # -------------------- 输出目录 --------------------

        # 所有实验结果的根目录
        #
        # 最终目录通常为：
        # runs/train/<name>/
        project="runs/train",

        # 当前实验名称
        #
        # 例如：
        # runs/train/yolo26n_seg_baseline/
        #
        # 后续消融实验可以命名为：
        # yolo26n_seg_p2
        # yolo26n_seg_attention
        # yolo26n_seg_lightneck
        # 等
        name=run_name,

        # -------------------- 权重保存 --------------------

        # 是否按固定 Epoch 间隔额外保存 checkpoint
        #
        # save_period=-1：
        # 不额外保存 epoch10.pt、epoch20.pt 等周期性权重
        #
        # 训练过程中仍会正常生成：
        # weights/best.pt
        # weights/last.pt
        #
        # 对普通论文训练来说，best.pt 和 last.pt 基本已经足够
        save_period=-1,

        # -------------------- 日志输出 --------------------

        # 是否在控制台显示详细训练信息
        # 包括 loss、显存占用、迭代进度等
        verbose=True,
    )