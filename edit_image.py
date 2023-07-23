#@title Edit Image

def process_image(image, method,  edit, factor):
    cfg = utils.args_to_cfg(method, edit, factor)
    image = image_processor.preprocess_image(image, is_batch=False)
    image = styleres.edit_images(image, cfg)
    image = image_processor.postprocess_image(image.detach().cpu().numpy(), is_batch=False).resize((256,256))
    return image

edited_image = process_image(input_image, method_dropdown.value, edits_dropdown.value, factor_slider.value)
total_width = input_image.width + edited_image.width
max_height = max(input_image.height, edited_image.height)
result_image = Image.new('RGB', (total_width, max_height))
result_image.paste(input_image, (0,0))
result_image.paste(edited_image, (input_image.width,0))
display(result_image)