from __future__ import absolute_import, print_function

import gradio as gr
import torch
import torch.nn as nn
from render_mk2 import Render_Class, module_fill
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Update_Slider:
    def __init__(self, module_dict):
        self.module_dict = module_dict

    def __call__(self, layer_name):
        if isinstance(self.module_dict[layer_name], nn.Linear):
            max_channel_num = self.module_dict[layer_name].out_features - 1
        else:
            max_channel_num = self.module_dict[layer_name].out_channels - 1
        return gr.update(maximum=max_channel_num)


def main():
    """Interactive gradio wrapper."""
    model = models.resnet34(weights=None)
    in_features = model.fc.in_features
    # Always reshape the last layer of any imported model to accomodate
    # dataset classes.
    model.fc = nn.Linear(in_features, 10)
    model.load_state_dict(
        torch.load(
            "C://Users//Noel//Documents//THESIS"
            "//Feature Visualization//Weights"
            "//resnet34_torchvision"
            "//test72_epoch446.pth"
        )
    )
    model.to(device).eval()
    module_dict = module_fill(model)

    # Define render object.
    render = Render_Class(model)
    update = Update_Slider(module_dict)

    with gr.Blocks() as demo:
        gr.Markdown("Visualize a variety of objectives.", visible=True)
        list_of_objectives = [
            "DeepDream",
            "Channel",
            "Neuron",
            "Interpolation",
            "Joint",
            "Diversity",
            "WRT Classes",
        ]
        inputs = {}
        output = {}
        buttons = {}
        with gr.Tabs():
            for objective_type in list_of_objectives:
                with gr.Tab(objective_type):
                    # type = gr.Markdown(Tab.label, visible=False)
                    type = gr.Markdown(
                        objective_type, visible=False
                    )  # jankiest of solutions but alas...
                    parameterization = gr.Radio(
                        choices=["fft", "pixel"], value="fft",
                        label="Parameterization"
                    )
                    threshold = gr.Slider(
                        64, 2048, step=64, label="Number of Iterations"
                    )
                    layer_selection = gr.Radio(
                        choices=list(module_dict.keys()), label="Layer"
                    )
                    # Objective class Channel or Neuron
                    if (
                        objective_type == list_of_objectives[1]
                        or objective_type == list_of_objectives[2]
                    ):
                        channel_selection = gr.Slider(
                            0, 511, step=1, label="Channel Number"
                        )
                        image_shape = gr.Number(1, precision=0, visible=False)
                        operator = gr.Radio(
                            choices=["Negative", "Positive"],
                            label="Available Operators",
                        )
                        inputs[objective_type] = [
                            type,
                            operator,
                            layer_selection,
                            channel_selection,
                            parameterization,
                            threshold,
                            image_shape,
                        ]
                    # Objective class Interpolation or Joint Activation
                    elif (
                        objective_type == list_of_objectives[3]
                        or objective_type == list_of_objectives[4]
                    ):
                        layer_selection_2 = gr.Radio(
                            choices=list(module_dict.keys()),
                            label="Second layer"
                        )
                        channel_selection = gr.Slider(
                            0, 511, step=1, label="Channel Number"
                        )
                        channel_selection_2 = gr.Slider(
                            0, 511, step=1, label="Second Channel Number"
                        )
                        image_shape = gr.Slider(
                            1, 10, value=1, step=1, label="Images to Produce"
                        )
                        layer_selection_2.change(
                            fn=update,
                            inputs=layer_selection_2,
                            outputs=channel_selection_2,
                        )
                        if objective_type == "Joint":
                            operator = gr.Radio(
                                choices=["+", "-"], label="Available Operators"
                            )
                        else:
                            operator = gr.Radio(
                                choices=[], label="Available Operators",
                                visible=False
                            )
                        inputs[objective_type] = [
                            type,
                            operator,
                            layer_selection,
                            channel_selection,
                            parameterization,
                            threshold,
                            image_shape,
                            layer_selection_2,
                            channel_selection_2,
                        ]
                    # Objective class Diversity
                    elif objective_type == list_of_objectives[5]:
                        channel_selection = gr.Slider(
                            0, 511, step=1, label="Channel Number"
                        )
                        image_shape = gr.Slider(
                            4, 10, step=2, label="Images to Produce"
                        )
                        operator = gr.Radio(
                            choices=[], label="Available Operators",
                            visible=False
                        )
                        inputs[objective_type] = [
                            type,
                            operator,
                            layer_selection,
                            channel_selection,
                            parameterization,
                            threshold,
                            image_shape,
                        ]
                    # Objective class WRT Classes
                    elif objective_type == list_of_objectives[6]:
                        channel_selection = gr.Slider(
                            0, 511, step=1, label="Channel Number"
                        )
                        image_shape = gr.Number(10, precision=0, visible=False)
                        operator = gr.Radio(
                            choices=[], label="Available Operators",
                            visible=False
                        )
                        inputs[objective_type] = [
                            type,
                            operator,
                            layer_selection,
                            channel_selection,
                            parameterization,
                            threshold,
                            image_shape,
                        ]
                    # Objective class DeepDream
                    else:
                        channel_selection = gr.Slider(
                            0, label="Channel Number", visible=False
                        )
                        image_shape = gr.Number(1, precision=0, visible=False)
                        operator = gr.Radio(
                            choices=[], label="Available Operators",
                            visible=False
                        )
                        inputs[objective_type] = [
                            type,
                            operator,
                            layer_selection,
                            channel_selection,
                            parameterization,
                            threshold,
                            image_shape,
                        ]
                    """ Check out the Huggingface introduction
                        to gradio blocks."""
                    layer_selection.change(
                        fn=update, inputs=layer_selection,
                        outputs=channel_selection
                    )
                    buttons[objective_type] = gr.Button("Create")
                    output[objective_type] = gr.Image().style(height=224)
                    start = buttons[objective_type].click(
                        render.render, inputs[objective_type],
                        output[objective_type]
                    )

                    stop = gr.Button("Abort")
                    stop.click(
                        fn=render.set_flag, inputs=None,
                        outputs=None, cancels=[start]
                    )
                    # TODO: Check if cancel can interrupt a function execution

    demo.queue(concurrency_count=1, max_size=1).launch()


if __name__ == "__main__":
    main()
