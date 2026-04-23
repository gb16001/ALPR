import torch
# test modual

import Dan_furnace
def train_Danlu_script(conf_file= 'configs/args-train.yaml'):
    trainer=Dan_furnace.Trainer_a_conf(conf_file)
    trainer.train()

def train_Danlu_scripts(cfg_list=[
    'configs/args-train.yaml',
    ]):
    for conf_file in cfg_list:
        print(f"Starting training with config: {conf_file}")
        trainer=Dan_furnace.Trainer_a_conf(conf_file)
        trainer.train()

        print(f"Finishing training for {conf_file}. Clearing memory...")
        # 删除 trainer 对象引用，确保其占用的 tensor 可以被回收
        del trainer
        import gc
        gc.collect()
        # 清理 PyTorch 缓存的显存，将其返回给 GPU
        torch.compiler.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Memory cleared for the next run.\n")

        

    return

def eval_Danlu(conf_file='configs/test.yaml'):
    # 'configs/test.yaml'
    results=[]
    evaluator=Dan_furnace.Eval_a_conf(conf_file)
    evaluator.test()
    if getattr(evaluator.args.val_set,'csvPaths',None) is None:
        print("No csvPaths found in config, skipping multiple dataset evaluation.")
        return
    for testSet in evaluator.args.val_set.csvPaths:
        print(f'testing {testSet}')
        evaluator.change_dataset(testSet)
        results.append(evaluator.test()) 
    # 遍历results列表，逐个元素换行打印
    for res in results:
        print(res)
    return

def run_speed_test(
    model_cls=None,       # 模型类，默认使用 mtLPRtr_CvNxt_FPN_STN_co2D_MAE_up
    img_size=(1080, 1920),# 输入图像尺寸 (H, W)
    batch_size=1,
    warmup_steps=20,
    test_steps=100,
    use_compile=True,     # 是否用 torch.compile 加速
    precision="fp32",     # 推理精度: "fp32" | "fp16" | "bf16"
):
    import torch
    import numpy as np
    from datasets import Dataset_rand, BatchBox

    # --- 精度参数校验 ---
    _PREC_MAP = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in _PREC_MAP:
        raise ValueError(f"precision 必须是 {list(_PREC_MAP)} 之一，got '{precision}'")
    amp_dtype = _PREC_MAP[precision]

    # --- 默认模型 ---
    if model_cls is None:
        from models.fullModel import mtLPRtr_CvNxt_FPN_STN_co2D_MAE_up as model_cls

    # --- 环境准备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在设备上运行: {device}")
    if precision in ("fp16", "bf16") and device.type != "cuda":
        print(f"警告: {precision} autocast 在 CPU 上无加速意义，回退到 fp32")
        precision = "fp32"
        amp_dtype = torch.float32

    # --- 初始化模型 ---
    print(f"正在初始化模型: {model_cls.__name__} ...")
    model = model_cls()
    model.to(device)
    model.eval()

    # --- torch.compile ---
    if use_compile:
        print("正在 torch.compile(mode='reduce-overhead') ...")
        model = torch.compile(model, mode="reduce-overhead")

    # --- 生成模拟输入 ---
    print(f"生成随机测试数据 (batch={batch_size}, img_size={img_size})...")
    imgs, targets = Dataset_rand.batch_rand(batchSize=batch_size, imgSize=img_size)
    dummy_batch = BatchBox(
        images=imgs,
        hr_LP_img=imgs,
        plateType=targets["plateType"],
        bboxes=targets["boxes"],
        LPs=targets["LPs"],
        LPs_delay=targets["LPs_delay"],
        verteces=targets["verteces"],
        theta=targets["theta"],
    )
    dummy_batch.mv_to_device(device)

    # autocast 上下文：fp32 时用 nullcontext 不引入任何开销
    from contextlib import nullcontext
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if precision != "fp32"
        else nullcontext()
    )

    def _forward():
        with amp_ctx:
            return model.forward(dummy_batch, need_stn=True, need_mae=False)

    # --- 开始预热（compile 后第一次编译 kernel，需要更多预热）---
    actual_warmup = warmup_steps * 2 if use_compile else warmup_steps
    print(f"开始预热 ({actual_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(actual_warmup):
            _forward()

    # --- 正式测量 ---
    print(f"正式统计速度 ({test_steps} iterations)...")
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    timings = np.zeros(test_steps)

    with torch.no_grad():
        for i in range(test_steps):
            starter.record()
            _forward()
            ender.record()
            # 同步 GPU，确保计算完成再统计时间
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender)  # 毫秒

    # --- 结果计算 ---
    avg_ms = np.mean(timings)
    std_ms = np.std(timings)
    fps = 1000.0 / avg_ms * batch_size

    compile_tag = "compiled" if use_compile else "eager"
    print("\n" + "=" * 40)
    print(f"推理速度报告 [{model_cls.__name__}] [{compile_tag}] [{precision}]")
    print(f"  图像尺寸:   {img_size[0]}x{img_size[1]},  BatchSize={batch_size}")
    print(f"  平均耗时:   {avg_ms:.3f} ms/batch")
    print(f"  标准差:     {std_ms:.3f} ms")
    print(f"  吞吐量:     {fps:.2f} FPS")
    print("=" * 40)

if __name__ == "__main__":
    # from models.fullModel import mtLPRtr_CvNxt_FPN_STN_co2D
    # run_speed_test(
    #     model_cls=mtLPRtr_CvNxt_FPN_STN_co2D,
    #     img_size=(540, 960), #(1160,720),
    #     batch_size=1,
    #     use_compile=True,
    #     precision="bf16"
    # )
    eval_Danlu('configs/CCPD-ab_backbone_convnext_pico.yaml')
    cfg_files=[
        'configs/CCPD-ab_backbone_convnext_pico.yaml',
        'configs/CCPD-mtLPRtr_CvNxt_FPN_STN_co2D.yaml',
        'configs/CCPD-mtLPRtr_CvNxtnano_FPN_STN_co2D.yaml',
        'configs/CCPD-exDETR_CDN_CvNxt_FPN_STN_co2D.yaml',
        'configs/CCPD18-ab_backbone_convnext_pico.yaml',
        'configs/CCPD18-mtLPRtr_CvNxt_FPN_STN_co2D.yaml',
        'configs/CCPD18-mtLPRtr_CvNxtnano_FPN_STN_co2D.yaml',
        'configs/CCPD18-exDETR_CDN_CvNxt_FPN_STN_co2D.yaml',
        ]
    # train_Danlu_scripts(cfg_list=cfg_files)
    
    pass
