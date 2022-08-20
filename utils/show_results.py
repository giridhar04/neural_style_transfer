from PIL import Image
from run_style_transfer import run_style_transfer

def show_results(content_path, style_path, num_iterations):

    best_img, best_loss = run_style_transfer(content_path , style_path , num_iterations = num_iterations)

    name = content_path.split('/')[-1].split('.')[0] + "_" + style_path.split('/')[-1].split('.')[0]
    path = 'static/uploads/' + name + "_styled.jpg"

    image2 = Image.fromarray(best_img)
    image2.save(path)

    return name + "_styled.jpg"
