from __future__ import absolute_import, print_function

import gradio as gr
import torch
import torchvision

from render_mk5 import Render_Class
from torchvision.models import list_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Cancels:
    """Handles all abort operations."""

    def __init__(self, dict_type: dict) -> None:
        self.ren_inst_dict = dict_type

    def abort_all_ops(self):
        """Sequentially changes the abort flag of
        all render class instances."""
        print("Operations Aborted!")
        for instance in self.ren_inst_dict.values():
            instance.abort_operation()


def check_if_cnn(listed_models):
    """Removes all vision transformers from the
    list of available models. Rendering does not
    work with them."""
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


def show_saving_fields(choice):
    if choice.strip("<p>\n/").lower() == "yes":
        return [gr.Textbox.update(visible=True), gr.Textbox.update(visible=True)]
    else:
        return [gr.Textbox.update(visible=False), gr.Textbox.update(visible=False)]


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
        activation_func = gr.Button(value="Enforce Leaky ReLU", visible=True)
        model_selection.change(
            fn=show_act_f_change, inputs=[model_selection], outputs=[activation_func]
        )

        upload = gr.UploadButton(
            label="Upload appropriate .pth file.",
            file_types=["pth"],
            file_count="single",
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
        # functions of class instances. Will make it much easier to retain
        # and modify information outside of gradio components.
        inputs = {}
        output = {}
        start_buttons = {}
        render_instances = {}
        tabs = {}
        with gr.Tabs():
            for objective_type in list_of_objectives:
                with gr.Tab(objective_type) as tabs[objective_type]:
                    # Initialize Render_Class instance for each tab.
                    render_instances[objective_type] = Render_Class()
                    print(
                        f"Initializing Instance for {objective_type} Visualization Objective"
                    )
                    # print(objective_type, render_instances[objective_type])
                    type = gr.Markdown(
                        objective_type, visible=False
                    )  # jankiest of solutions but alas...
                    parameterization = gr.Radio(
                        choices=["fft", "pixel"], value="fft", label="Parameterization"
                    )
                    retain = gr.Radio(
                        choices=["Yes", "No"], value="No", label="Save rendered output?"
                    )
                    saving_path = gr.Textbox(
                        label="Input absolute save path", visible=False
                    )
                    naming_scheme = gr.Textbox(
                        label="Input desired file name", visible=False
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
                    # CHANGES AND BUTTONS ------------------------------
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
                    retain.change(
                        fn=render_instances[objective_type].save_output,
                        inputs=[retain],
                        outputs=None,
                    )
                    retain.change(
                        fn=show_saving_fields,
                        inputs=[retain],
                        outputs=[saving_path, naming_scheme],
                    )
                    saving_path.submit(
                        fn=render_instances[objective_type].where_to_save,
                        inputs=[saving_path],
                        outputs=None,
                    )
                    naming_scheme.submit(
                        fn=render_instances[objective_type].how_to_name,
                        inputs=[naming_scheme],
                        outputs=None,
                    )
                    # Make Buttons ---------------------------------------
                    start_buttons[objective_type] = gr.Button("Create")
                    output[objective_type] = gr.Image().style(height=224)
                    # Start Button trigger
                    start = start_buttons[objective_type].click(
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
        # Each time a user selects a different tab, all previous operations
        # cancel as a means of controlling available cuda or system memory.
        # Unfortunately, rendering on multiple tabs is a very taxing operation.
        for objective_type in list_of_objectives:
            # Make object in order to pass the render_instances dictionary
            cancel_ops = Cancels(render_instances)
            # Create actions for each Tab's selection.
            tabs[objective_type].select(
                fn=cancel_ops.abort_all_ops,
                inputs=None,
                outputs=None,
                cancels=None
            )
    # Set concurency N=number of objective tabs
    # Set size n=number changes needed
    print("Starting server...")
    demo.queue(concurrency_count=6, max_size=10).launch()


if __name__ == "__main__":
    main()
