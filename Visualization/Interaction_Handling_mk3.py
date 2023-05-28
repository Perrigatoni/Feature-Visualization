from __future__ import absolute_import, print_function

import gradio as gr
import torch

from render_mk3 import Render_Class
from torchvision.models import list_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    """Interactive gradio wrapper.

        The docs are not very helpful regarding complex functionality
        from a single function. 

        This interface is for simple display purposes only.
        Do try the standalone functions to experiment with
        mixing, more objectives, more lax parameterization.
    """
    with gr.Blocks() as demo:
        gr.Markdown(
            """<H1 style="text-align: center;">Visualize a variety of objectives.</H1>""",
            visible=True,
        )
        gr.Markdown(
            """<H3 style="text-align: center;">Select layers and drag sliders to check out different channels, neurons etc.</H3>""",
            visible=True,
        )

        list_of_objectives = [
            "DeepDream",
            "Channel",
            "Neuron",
            "Interpolation",
            "Joint",
            "Diversity",
            # "WRT Classes",
        ]
        # Make dictionaries of I/O and Buttons to have seperate calls to
        # functions class objects. Will make it much easier to retain
        # information.
        inputs = {}
        output = {}
        buttons = {}
        with gr.Tabs():
            for objective_type in list_of_objectives:
                with gr.Tab(objective_type):
                    # Initialize render object.
                    render = Render_Class(
                        change_act_func=False,
                    )
                    type = gr.Markdown(
                        objective_type, visible=False
                    )  # jankiest of solutions but alas...

                    model_list = list_models()
                    model_selection = gr.Radio(
                        choices=model_list, label="Available Torchvision Models"
                    )

                    parameterization = gr.Radio(
                        choices=["fft", "pixel"], value="fft", label="Parameterization"
                    )
                    threshold = gr.Slider(
                        0, 1024, step=16, label="Number of Iterations"
                    )
                    layer_selection = gr.Radio(choices=[], label="Layer")
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
                        layer_selection_2 = gr.Radio(choices=[], label="Second layer")
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
                            fn=render.update_sliders,
                            inputs=layer_selection_2,
                            outputs=channel_selection_2,
                        )
                        if objective_type == "Joint":
                            operator = gr.Radio(
                                choices=["+", "-"], label="Available Operators"
                            )
                        else:
                            operator = gr.Radio(
                                choices=[], label="Available Operators", visible=False
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
                        model_selection.change(
                            fn=render.available_layers,
                            inputs=model_selection,
                            outputs=layer_selection_2,
                        )
                    # Objective class Diversity
                    elif objective_type == list_of_objectives[5]:
                        channel_selection = gr.Slider(
                            0, 511, step=1, label="Channel Number"
                        )
                        image_shape = gr.Slider(
                            4, 10, step=2, label="Images to Produce"
                        )
                        operator = gr.Radio(
                            choices=[], label="Available Operators", visible=False
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
                    # elif objective_type == list_of_objectives[6]:
                    #     channel_selection = gr.Slider(
                    #         0, 511, step=1, label="Channel Number"
                    #     )
                    #     image_shape = gr.Number(10, precision=0, visible=False)
                    #     operator = gr.Radio(
                    #         choices=[], label="Available Operators", visible=False
                    #     )
                    #     inputs[objective_type] = [
                    #         type,
                    #         operator,
                    #         layer_selection,
                    #         channel_selection,
                    #         parameterization,
                    #         threshold,
                    #         image_shape,
                    #     ]
                    # Objective class DeepDream
                    else:
                        channel_selection = gr.Slider(
                            0, label="Channel Number", visible=False
                        )
                        image_shape = gr.Number(1, precision=0, visible=False)
                        operator = gr.Radio(
                            choices=[], label="Available Operators", visible=False
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
                    model_selection.change(
                        fn=render.available_layers,
                        inputs=model_selection,
                        outputs=layer_selection,
                    )
                    layer_selection.change(
                        fn=render.update_sliders,
                        inputs=layer_selection,
                        outputs=channel_selection,
                    )
                    buttons[objective_type] = gr.Button("Create")
                    output[objective_type] = gr.Image().style(height=224)
                    start = buttons[objective_type].click(
                        render.render,
                        inputs[objective_type],
                        output[objective_type]
                    )

                    stop = gr.Button("Abort")
                    stop.click(
                        fn=render.abort_operation,
                        inputs=None,
                        outputs=None,
                        cancels=[start]
                    )

    demo.queue(concurrency_count=2, max_size=2).launch()


if __name__ == "__main__":
    main()
