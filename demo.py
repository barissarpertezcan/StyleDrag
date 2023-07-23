from PIL import Image
import torch
import os
# from IPython.display import display

from utils import AppUtils
utils = AppUtils()

from inference import initialize_styleres
from datasets.process_image import ImageProcessor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
styleres = initialize_styleres('checkpoints/styleres_ffhq.pth', device)
image_processor = ImageProcessor()
if device == 'cuda':
  print("StyleGAN compiles custom C++ codes, therefore, the first run might be slow.")

selected_image = '116.jpg' #@param ["116.jpg", "11654.jpg", "1219.jpg", "3897.jpg", "6898.jpg", "756.jpg"]
input_image = Image.open(f"samples/inference_samples/{selected_image}")

def process_image(image, method,  edit, factor):
    cfg = utils.args_to_cfg(method, edit, factor)
    image = image_processor.preprocess_image(image, is_batch=False)
    image = styleres.edit_images(image, cfg)
    image = image_processor.postprocess_image(image.detach().cpu().numpy(), is_batch=False).resize((256,256))
    return image


method_dropdown = "StyleClip"
edits_dropdown = "Purple Hair"
factor_slider = 0.1

edited_image = process_image(input_image, method_dropdown, edits_dropdown, factor_slider) # method_dropdown.value, edits_dropdown.value, factor_slider.value
total_width = input_image.width + edited_image.width
max_height = max(input_image.height, edited_image.height)
result_image = Image.new('RGB', (total_width, max_height))
result_image.paste(input_image, (0,0))
result_image.paste(edited_image, (input_image.width,0))
result_image.show()
# display(result_image)