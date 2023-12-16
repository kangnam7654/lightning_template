from pathlib import Path
import cv2
import numpy as np
import torch
import gradio as gr
from torchvision.utils import make_grid

from mvface_packages.stylegan2.model import Generator
from facenet_pytorch import MTCNN


class GradioInferencer:
    def __init__(self):
        self.model = self.get_default_model()
        self.page = self.define_page()

    def get_default_model(self, size=1024, style_dim=512, n_mlp=8, eval=True):
        model = Generator(size=size, style_dim=style_dim, n_mlp=n_mlp)
        if eval:
            model = model.eval()
            model = model.cuda()
        return model

    def apply_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["g_ema"])

    def define_page(self):
        with gr.Blocks() as page:
            with gr.Row():
                generated = gr.Image(label="Generated Image")
            generate_button = gr.Button("Generate")

            with gr.Row():
                with gr.Column():
                    to_show = ["No weight", "FFHQ", "METAWORLD"]
                    weight_index = gr.Dropdown(
                        choices=to_show, label="Preset Weight", type="index"
                    )
                with gr.Column():
                    weight_file = gr.File(label="Weight")
            weight_change_button = gr.Button("Weight Change")

            generate_button.click(fn=self.inference, outputs=generated)
            weight_change_button.click(
                fn=self.change_weight,
                inputs=[weight_file, weight_index],
                outputs=None,
            )

        return page

    def _preset_weights(self):
        weight_dir = Path(__file__).parent.joinpath("weights")
        weights = [
            None,
            str(weight_dir.joinpath("stylegan2-ffhq-config-f.pt")),
            str(weight_dir.joinpath("finetune_231114.pt")),
        ]
        return weights

    def change_weight(self, weight_file=None, weight_index=None):
        preset_weights = self._preset_weights()
        if weight_index is not None:
            to_load = preset_weights[weight_index]
        elif weight_file is not None:
            to_load = weight_file.orig_name
        else:
            to_load = ""
        state_dict = torch.load(to_load)["g_ema"]
        self.model.load_state_dict(state_dict)

    def invert(self, tensor_image) -> np.ndarray:
        image = tensor_image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.cpu().detach().numpy()
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        image = np.clip(image, -1, 1)
        image = (image + 1) / 2
        return image

    def inference(self, latent=None):
        if latent is None:
            latent = torch.randn(1, 512).cuda()
        output = self.model([latent])[0]
        image = self.invert(output)
        return image


def main():
    inferencer = GradioInferencer()
    page = inferencer.define_page()
    page.launch(share=True)


if __name__ == "__main__":
    main()
