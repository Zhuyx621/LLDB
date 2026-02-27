import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset

############################################################
# Options
############################################################
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True)
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

torch.backends.cudnn.benchmark = True

############################################################
# Logger
############################################################
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if "pretrain_model" not in key and "resume" not in key
    )
)

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)

logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

############################################################
# Dataset
############################################################
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(f"Number of images in [{dataset_opt['name']}]: {len(test_set)}")
    test_loaders.append(test_loader)

############################################################
# Model
############################################################
model = create_model(opt)
device = model.device

sde = util.GOUB(
    lambda_square=opt["sde"]["lambda_square"],
    T=opt["sde"]["T"],
    schedule=opt["sde"]["schedule"],
    eps=opt["sde"]["eps"],
    device=device,
)

sde.set_model(model.model)

############################################################
# Enhancement
############################################################

MAX_SIZE = 512   # ⭐ 6GB安全尺寸（384更稳，512质量更好）

total_processed = 0
total_skipped = 0
total_failed = 0

for test_loader in test_loaders:

    test_set_name = test_loader.dataset.opt["name"]
    logger.info(f"\nTesting [{test_set_name}]...")

    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_times = []

    for i, test_data in enumerate(test_loader):

        try:

            img_path = test_data["LQ_path"][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_img_path = os.path.join(dataset_dir, img_name + ".png")

            # ==============================
            # 断点续跑
            # ==============================
            if os.path.exists(save_img_path):
                logger.info(f"Skip (exists): {img_name}")
                total_skipped += 1
                continue

            LQ = test_data["LQ"]

            # ==============================
            # 强制通道修复（核心）
            # ==============================
            if LQ.size(1) == 1:
                LQ = LQ.repeat(1, 3, 1, 1)

            elif LQ.size(1) == 2:
                LQ = torch.cat([LQ, LQ[:, :1, :, :]], dim=1)

            elif LQ.size(1) == 4:
                LQ = LQ[:, :3, :, :]

            # ==============================
            # 记录原始尺寸
            # ==============================
            _, _, orig_h, orig_w = LQ.size()
            resized = False

            # ==============================
            # 防OOM自动缩放
            # ==============================
            if max(orig_h, orig_w) > MAX_SIZE:
                resized = True
                scale = MAX_SIZE / max(orig_h, orig_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                LQ = F.interpolate(
                    LQ,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )

            # ==============================
            # 无GT增强模式
            # ==============================
            model.feed_data(LQ, LQ, LQ)

            tic = time.time()
            with torch.no_grad():
                model.test(sde, save_states=False)
            toc = time.time()

            test_times.append(toc - tic)

            visuals = model.get_current_visuals()
            SR_img = visuals["Output"]

            if SR_img.dim() == 3:
                SR_img = SR_img.unsqueeze(0)

            # 恢复原始尺寸
            if resized:
                SR_img = F.interpolate(
                    SR_img,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )

            output = util.tensor2img(SR_img.squeeze())
            util.save_img(output, save_img_path)

            logger.info(f"Saved: {img_name}")
            total_processed += 1

            # 显存清理
            del LQ, SR_img, visuals, output
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed: {img_name} | Error: {str(e)}")
            total_failed += 1
            torch.cuda.empty_cache()
            continue

    if test_times:
        print(f"Average time: {np.mean(test_times):.4f}s")

print("\n====================================")
print(f"Total processed : {total_processed}")
print(f"Total skipped   : {total_skipped}")
print(f"Total failed    : {total_failed}")
print("Enhancement finished safely.")
print("====================================")
