from pipelines.gan import GANPipeline
from models.encoder import Encoder
from models.decoder import Decoder


def main():
    discriminator = Encoder()
    generator = Decoder()
    gan = GANPipeline(
        generator=generator,
        discriminator=discriminator,
        manual_ckpt_save_path="./gan.ckpt",
        lr_d=1e-3,
        lr_g=1e-3,
        n_critic=1,
        image_logging_interval=100,
    )
    
    

if __name__ == "__main__":
    main()
