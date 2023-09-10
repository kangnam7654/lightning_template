import os
import argparse
import pytorch_lightning as pl
import torch
from datamodules.animegan_datamodule import AnimeGANDatamodule
from models.animegan import AnimeDiscriminator, Generator
from pipelines.animegan_pipeline import AnimeGANPipeline
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from kangnam_packages.datamodules.lightning_wrapper import LightningDataWrapper
from models.animegan import AnimeDiscriminator

# Model load

torch.set_float32_matmul_precision("high")
pl.seed_everything(2023)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image_dir", type=str)
    parser.add_argument("--style_image_dir", type=str)
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--image_logging_interval", type=int, default=100)
    parser.add_argument("--data_length", type=int, default=None)
    parser.add_argument("--data_repeat", type=int, default=1)
    parser.add_argument("--pretraining_epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument(
        "--w_adv", type=float, default=1, help="weight adversarial loss"
    )
    parser.add_argument(
        "--w_con", type=float, default=0.033, help="weight constuction loss"
    )
    parser.add_argument("--w_gra", type=float, default=0.033, help="weight gray loss")
    parser.add_argument("--w_col", type=float, default=0.0002, help="weihgt color loss")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--accumulate_interval", type=int, default=4)
    args = parser.parse_args()
    return args


def main(args):
    # CONFIGS
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    resolution = args.resolution
    lr = args.lr
    w_adv = args.w_adv
    w_con = args.w_con
    w_gra = args.w_gra
    w_col = args.w_col
    image_logging_interval = args.image_logging_interval
    data_length = args.data_length
    data_repeat = args.data_repeat
    show = args.show
    pretraining_epoch = args.pretraining_epoch

    # ===============================
    # | Content Data directory Load |
    # ===============================
    if args.content_image_dir is not None:
        content_image_dir = args.content_image_dir
    else:
        home_dir = os.path.expanduser("~")
        storage = "link_870_4T_1"  # for server
        paths = [home_dir, storage, "datasets", "animegan_training", "real"]

        content_image_dir = os.path.join(*paths)
        if os.path.isdir(content_image_dir) is False:
            paths[1] = "hdd"
            content_image_dir = os.path.join(*paths)
            assert os.path.isdir(content_image_dir), "--content_image_dir 인자를 확인하세요."
        print("--content_image_dir 인자가 설정 되지 않았습니다. 미리 설정된 경로를 불러옵니다.")
    print(f"불러온 content_image_dir : {content_image_dir}")

    # | Style Data Directory Load |
    if args.style_image_dir is not None:
        style_image_dir = args.style_image_dir
    else:
        home_dir = os.path.expanduser("~")
        storage = "link_870_4T_1"  # for server
        paths = [home_dir, storage, "datasets", "animegan_training", "unreal"]
        style_image_dir = os.path.join(*paths)

        if os.path.isdir(style_image_dir) is False:
            paths[1] = "hdd"
            style_image_dir = os.path.join(*paths)
            assert os.path.isdir(content_image_dir), "--style_image_dir 인자를 확인하세요."
        print("--style_image_dir 인자가 설정 되지 않았습니다. 미리 설정된 경로를 불러옵니다.")
    print(f"불러온 sytle_image_dir : {style_image_dir}")

    # =============
    # | 모델 Load |
    # =============
    generator = Generator()
    state_dict = torch.load(
        # "src/animegan/weights/celeba_distill.pt"
        "src/animegan/weights/face_paint_512_v1.pt"
        # "src/animegan/weights/face_paint_512_v2.pt"
    )
    generator.load_state_dict(state_dict)
    # discriminator = AnimeDiscriminator()
    discriminator = Discriminator(256)

    pipeline = AnimeGANPipeline(
        generator=generator,
        discriminator=discriminator,
        lr=lr,
        manual_ckpt_save_path="./checkpoints/animegan_v1.ckpt",
        show=show,
        image_logging_interval=image_logging_interval,
        pretraining_epoch=pretraining_epoch,
        w_adv=w_adv,
        w_con=w_con,
        w_gra=w_gra,
        w_col=w_col,
        accumulate_interval=args.accumulate_interval,
    )

    # = Logger 설정 =
    wandb_logger = WandbLogger(name="AnimeGAN_v1", project="AnimeGAN")
    wandb_logger.watch(pipeline, log="all")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=5000,
        dirpath="./checkpoints",
        filename="animegan_v1_",
        verbose=True,
        save_last=True,
    )

    # ===================
    # | Trainer 불러오기 |
    # ===================
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=args.log_every_n_steps,
        # precision="16-mixed",
    )
    # =======================
    # = Dataset 설정 및 래핑 =
    # =======================
    dataset = AnimeGANDatamodule(
        content_image_dir=content_image_dir,
        style_image_dir=style_image_dir,
        resolution=resolution,
        data_length=data_length,
        data_repeat=data_repeat,
    )
    wrapped_module = LightningDataWrapper(
        dataset=dataset, batch_size=batch_size, num_workers=args.num_workers
    )

    # 달려!
    trainer.fit(model=pipeline, datamodule=wrapped_module)


if __name__ == "__main__":
    args = parse_args()
    main(args)
