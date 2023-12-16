from pathlib import Path
import pytorch_lightning as pl
import torch
from pipelines.stylegan2_pipeline import StyleGAN2Pipeline
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dataset_modules.stylegan2_datamodule import StyleGan2Datamodule
from mvface_packages.stylegan2_nvidia.training.networks import Discriminator, Generator
from mvface_packages.dataset_modules.lightning_wrapper import LightningDataWrapper
from mvface_packages.parser import get_parser

torch.set_float32_matmul_precision("high")
pl.seed_everything(2023)


def parse_arg():
    parser = get_parser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/kangnam/link_870_4T_1/datasets/animegan_training/unreal",
    )
    parser.add_argument("--weight", default=None)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--z_dim", type=int, default=512)
    parser.add_argument("--w_dim", type=int, default=512)
    parser.add_argument("--c_dim", type=int, default=0)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--ckpt_save_interval", type=int, default=10000)
    parser.add_argument("--image_logging_interval", type=int, default=100)
    parser.add_argument(
        "--checkpoint_save_path",
        type=str,
        default="./checkpoints/stylegan_231016.pt",
    )
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--return_label", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--logger_project", type=str, default="stylegan2")
    parser.add_argument("--logger_name", default="Stylegan2")
    parser.add_argument("--logger_save_dir", default=None)

    args = parser.parse_args()
    return args


def main(args):
    # ===========
    # | CONFIGS |
    # ===========
    batch_size = args.batch_size
    log_every_n_steps = 50
    data_repeat = args.data_repeat
    data_dir = args.data_dir

    # ==============
    # | Model Load |
    # ==============
    generator = Generator(
        z_dim=args.z_dim,
        c_dim=args.c_dim,
        w_dim=args.w_dim,
        img_resolution=args.resolution,
        img_channels=3,
    )

    if not args.from_scratch:
        if args.weight is None:
            state_dict = torch.load(
                Path(__file__).parent.joinpath("weights", "ffhq_convert.pt")
            )
        else:
            state_dict = torch.load(args.weight)
        generator.load_state_dict(state_dict)
    discriminator = Discriminator(
        c_dim=args.c_dim, img_resolution=args.resolution, img_channels=3
    )

    # | Pipelines |
    pipeline = StyleGAN2Pipeline(
        generator=generator,
        discriminator=discriminator,
        manual_ckpt_save_path=args.checkpoint_save_path,
        lr_d=args.lr_d,
        lr_g=args.lr_g,
        image_logging_interval=args.image_logging_interval,
        latent_dim=args.z_dim,
        show=args.show,
    )

    wandb_logger = WandbLogger(
        name=args.logger_name,
        save_dir=args.logger_save_dir,
        project=args.logger_project,
    )
    wandb_logger.watch(pipeline, log="all")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[lr_monitor, ckpt_callback],
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
    )

    # | Dataset and Wrapper |
    dataset = StyleGan2Datamodule(
        data_dir=data_dir,
        return_label=args.return_label,
        resolution=args.resolution,
        data_length=args.data_truncation,
        data_repeat=data_repeat,
    )
    wrapped_module = LightningDataWrapper(
        dataset=dataset, batch_size=batch_size, num_workers=8
    )

    # Run!
    trainer.fit(model=pipeline, datamodule=wrapped_module)


if __name__ == "__main__":
    args = parse_arg()
    args.resolution = 1024
    args.data_dir = "/home/kangnam/link_870_4T_1/datasets/animegan_training/unreal1024"
    args.batch_size = 2
    args.num_workers = 4
    main(args)
