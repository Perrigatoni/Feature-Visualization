from __future__ import absolute_import, print_function
import gradio as gr

import torch.nn as nn
from torchvision import models

from render_mk2 import render, module_fill


model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)
module_dict = module_fill(model)

def update_slider_max(layer_name):
    if isinstance(module_dict[layer_name], nn.Linear):
        max_channel_num = module_dict[layer_name].out_features - 1
    else:
        max_channel_num = module_dict[layer_name].out_channels - 1
    return gr.update(maximum=max_channel_num)   


def main():
    """ Main function meant to be wrapped with Gradio
        in order to create visual interface."""
    
    with gr.Blocks() as demo:
        gr.Markdown('Visualize a variety of objectives.', visible=True)
        # TODO: complete work for WRT Classes
        list_of_objectives = ['DeepDream',
                              'Channel',
                              'Neuron',
                              'Interpolation',
                              'Joint',
                              'Diversity',
                              'WRT Classes'   
                              ]
        inputs = {}
        output = {}
        buttons = {}
        with gr.Tabs():
            for objective_type in list_of_objectives:
                with gr.Tab(objective_type) as Tab:
                    # type = gr.Markdown(Tab.label, visible=False)
                    type = gr.Markdown(objective_type,
                                       visible=False)  # jankiest of solutions but alas...
                    parameterization = gr.Radio(choices=['fft', 'pixel'],
                                                value='fft',
                                                label='Parameterization')
                    threshold = gr.Slider(64,
                                          2048,
                                          step=64,
                                          label='Number of Iterations')
                    layer_selection = gr.Radio(choices=list(module_dict.keys()),
                                               label='Layer')
                    # Objective class Channel or Neuron
                    if objective_type == list_of_objectives[1] or objective_type == list_of_objectives[2]:
                        channel_selection = gr.Slider(0,
                                                      511,
                                                      step=1,
                                                      label='Channel Number')
                        image_shape = gr.Number(1,
                                                precision=0,
                                                visible=False)
                        operator = gr.Radio(choices=['Negative', 'Positive'],
                                            label='Available Operators')
                        inputs[objective_type] = [type,
                                                  operator,
                                                  layer_selection,
                                                  channel_selection,
                                                  parameterization,
                                                  threshold,
                                                  image_shape,
                                                  ]
                    # Objective class Interpolation or Joint Activation
                    elif objective_type == list_of_objectives[3] or objective_type == list_of_objectives[4]:
                        
                        layer_selection_2 = gr.Radio(choices=list(module_dict.keys()),
                                                     label='Second layer')
                        channel_selection = gr.Slider(0,
                                                      511,
                                                      step=1,
                                                      label='Channel Number')
                        channel_selection_2 = gr.Slider(0,
                                                      511,
                                                      step=1,
                                                      label='Second Channel Number')
                        image_shape = gr.Slider(1,
                                                10,
                                                value=1,
                                                step=1,
                                                label='Images to Produce')
                        layer_selection_2.change(fn=update_slider_max,
                                            inputs=layer_selection_2,
                                            outputs=channel_selection_2)
                        if objective_type == "Joint":
                            operator = gr.Radio(choices=['+', '-'],
                                                label='Available Operators')
                        else:
                            operator = gr.Radio(choices=[],
                                                label='Available Operators',
                                                visible=False)
                        inputs[objective_type] = [type,
                                                  operator,
                                                  layer_selection,
                                                  channel_selection,
                                                  parameterization,
                                                  threshold,
                                                  image_shape,
                                                  layer_selection_2,
                                                  channel_selection_2,
                                                  ]
                    #Objective class Diversity
                    elif objective_type == list_of_objectives[5]:
                        channel_selection = gr.Slider(0,
                                                      511,
                                                      step=1,
                                                      label='Channel Number')
                        image_shape = gr.Slider(4,
                                                10,
                                                step=2,
                                                label='Images to Produce')
                        operator = gr.Radio(choices=[],
                                            label='Available Operators',
                                            visible=False)
                        inputs[objective_type] = [type,
                                                  operator,
                                                  layer_selection,
                                                  channel_selection,
                                                  parameterization,
                                                  threshold,
                                                  image_shape,
                                                  ]
                    # Objective classes Channel Weight and Direction
                    else:
                        channel_selection = gr.Slider(0,
                                                      label='Channel Number',
                                                      visible=False)
                        image_shape = gr.Number(1,
                                                precision=0,
                                                visible=False)
                        operator = gr.Radio(choices=[],
                                            label='Available Operators',
                                            visible=False)
                        inputs[objective_type] = [type,
                                                  operator,
                                                  layer_selection,
                                                  channel_selection,
                                                  parameterization,
                                                  threshold,
                                                  image_shape,
                                                  ]
                    """ Check out the Huggingface introduction to gradio blocks."""
                    layer_selection.change(fn=update_slider_max,
                                            inputs=layer_selection,
                                            outputs=channel_selection)
                    buttons[objective_type] = gr.Button('Create')
                    output[objective_type] = gr.Image().style(height=224)
                    
                    buttons[objective_type].click(render,
                                                  inputs[objective_type],
                                                  output[objective_type])
                    
    demo.queue().launch()



    #     with gr.Tab('Channel Objective') as channel_tab:
    #         type = gr.Markdown(channel_tab.label, visible=False)  # jankiest of solutions but alas...
    #         layer_selection = gr.Radio(choices=list(module_dict.keys()), label='Layer')
    #         channel_selection = gr.Slider(0, 512, label='Channel Number')
    #         parameterization = gr.Radio(choices=['fft', 'pixel'], label='Parameterization')
    #         threshold = gr.Slider(64, 2048, step=64, label='Number of Iterations')
    #         image_shape = gr.Slider(1, 10, step=1, label='Images to Produce')
            
    #         channel_inputs = [type,
    #                             layer_selection,
    #                             channel_selection,
    #                             parameterization,
    #                             threshold,
    #                             image_shape
    #                             ]
    #         channel_button = gr.Button('Create')
            
    

    #     with gr.Tab('Neuron Objective') as neuron_tab:
    #         type = gr.Markdown(neuron_tab.label, visible=False)
    #         layer_selection = gr.Radio(choices=list(module_dict.keys()), label='Layer Selection')
    #         channel_selection = gr.Slider(0, 512, label='Channel Number')
    #         parameterization = gr.Radio(choices=['fft', 'pixel'], label='Parameterization')
    #         threshold = gr.Slider(64, 2048, step=64, label='Number of Iterations')
    #         image_shape = gr.Slider(1, 10, step=1, label='Images to Produce')

    #         neuron_inputs = [type,
    #                             layer_selection,
    #                             channel_selection,
    #                             parameterization,
    #                             threshold,
    #                             image_shape
    #                             ]
    #         neuron_button = gr.Button('Create')
            
    #     output = gr.Image().style(height=224)
        
    #     channel_button.click(render, channel_inputs, output)
    #     neuron_button.click(render, neuron_inputs, output)
    # demo.launch()



if __name__ == "__main__":
    main()