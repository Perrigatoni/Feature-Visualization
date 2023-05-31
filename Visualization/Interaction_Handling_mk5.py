from __future__ import absolute_import, print_function

import gradio as gr
import torch
import torchvision

from render_mk5 import Render_Class
from torchvision.models import list_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Cancels:
    def __init__(self, dict_type: dict) -> None:
        self.ren_inst_dict = dict_type
        print("Been through __init__")
        # del self.ren_inst_dict[name]
        print(self.ren_inst_dict)

    def abort_other_ops(self):
        for instance in self.ren_inst_dict.values():
            print("Been through abort other ops")
            instance.abort_operation()


def check_if_cnn(listed_models):
    cnn_list = []
    for name in listed_models:
        if "vit" in name or "swin" in name:
            pass
        else:
            cnn_list.append(name)
    return cnn_list


def show_act_f_change(model_name):
    """Makes user unable to select option Leaky ReLU
    if model is googlenet, since it has no named module
    for ReLU."""
    if model_name.strip("<p>\n/").lower() == "googlenet":
        return gr.Button.update(visible=False)
    elif model_name.strip("<p>\n/").lower() == "inception_v3":
        return gr.Button.update(visible=False)
    else:
        return gr.Button.update(visible=True)


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
            """<H1 style="text-align: center;">Visualize a
              variety of objectives.</H1>""",
            visible=True,
        )
        gr.Markdown(
            """<H3 style="text-align: center;">Select layers and drag sliders
              to check out different channels, neurons etc.</H3>""",
            visible=True,
        )
        model_list = check_if_cnn(list_models(module=torchvision.models))
        model_selection = gr.Radio(
            choices=model_list, label="Available Torchvision Models"
        )
        activation_func = gr.Button(value="Enforce Leaky ReLU",
                                    visible=True)
        model_selection.change(fn=show_act_f_change,
                               inputs=[model_selection],
                               outputs=[activation_func])
        upload = gr.UploadButton(label="Upload appropriate .pth file.",
                                 file_types=["pth"],
                                 file_count="single"
        )
        list_of_objectives = [
            "DeepDream",
            "Channel",
            "Neuron",
            "Interpolation",
            "Joint",
            "Diversity",
        ]
        # Make dictionaries of I/O and Buttons to have seperate calls to
        # functions class objects. Will make it much easier to retain
        # information.
        inputs = {}
        output = {}
        buttons = {}
        render_instances = {}
        tabs_dict = {}
        with gr.Tabs():
            for objective_type in list_of_objectives:
                with gr.Tab(objective_type) as tabs_dict[objective_type]:
                    # Initialize Render_Class instances.
                    render_instances[objective_type] = Render_Class()
                    print(objective_type, render_instances[objective_type])
                    type = gr.Markdown(
                        objective_type, visible=False
                    )  # jankiest of solutions but alas...
                    parameterization = gr.Radio(
                        choices=["fft", "pixel"],
                        value="fft",
                        label="Parameterization"
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
                        layer_selection_2 = gr.Radio(choices=[],
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
                            fn=render_instances[objective_type].update_sliders,
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
                        model_selection.change(
                            fn=render_instances[objective_type].available_layers,
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
                            choices=[],
                            label="Available Operators",
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
                            choices=[],
                            label="Available Operators",
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
                    # CHANGES AND BUTTONS
                    model_selection.change(
                        fn=render_instances[objective_type].available_layers,
                        inputs=model_selection,
                        outputs=layer_selection,
                    )
                    activation_func.click(
                        fn=render_instances[objective_type].handle_act_func,
                        inputs=None,
                        outputs=None,
                    )
                    layer_selection.change(
                        fn=render_instances[objective_type].update_sliders,
                        inputs=layer_selection,
                        outputs=channel_selection,
                    )
                    upload.upload(
                        fn=render_instances[objective_type].state_dict_upload,
                        inputs=[upload],
                        outputs=None,
                    )
                    # Make Buttons
                    buttons[objective_type] = gr.Button("Create")
                    output[objective_type] = gr.Image().style(height=224)
                    # Start Button trigger
                    start = buttons[objective_type].click(
                        render_instances[objective_type].render,
                        inputs[objective_type],
                        output[objective_type],
                    )
                    # Stop Button trigger
                    stop = gr.Button("Abort")
                    stop.click(
                        fn=render_instances[objective_type].abort_operation,
                        inputs=None,
                        outputs=None,
                        cancels=[start],
                    )
        
        for objective_type in list_of_objectives:
            cancel_ops = Cancels(render_instances)
            # cancel_others_dict[objective_type].remove_and_return()
            tabs_dict[objective_type].select(fn=cancel_ops.abort_other_ops,
                                             inputs=None,
                                             outputs=None,
                                             cancels=None)
    # Set concurency N=number of objective tabs (limit
    # to 2 or 3 if GPU memory =< 4Gb)
    # Set size n=number changes needed
    demo.queue(concurrency_count=6, max_size=10).launch()


if __name__ == "__main__":
    main()
