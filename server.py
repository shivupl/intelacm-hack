import gradio as gr
from PIL import Image

from background import process_image


def run(image, view):
    clean_img, result = process_image(image, view)
    return clean_img, result


demo = gr.Interface(
    fn = run,
    inputs = [gr.Image(type = "pil", label = "Upload Vehicle Image(One angle)"),
              gr.Radio(["front", "back", "side"], label="Select the angle in the photo (Only front currently available)")
            ],
    outputs = [
        gr.Image(label="Vehicle Image"),
        gr.Text(label="AI Detection Result"),
    ],
    title="AutoAuth",
    description = "Detects tamperered images"
   
)



if __name__ == "__main__":
    demo.launch(share = True)