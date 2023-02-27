# Feature-Visualization

This project was meant to recreate and study the visualizations produced by
the original Lucid Library and GoogLeNet with Imagenet dataset.

For this purpose, necessary files were created based on Lucid's original
conceptualization of their project taking into serious account ProGamerGov's
pytorch implementation of the same project (named Lucent), whose code proved
to be a strong basis for further understanding the code provided by the Lucid
creators, since this project was also based on pytorch.

Restricting the packages to better conform with torchvision's own implementations
and updating those deemed too slow/cumbersome to use was also a key objective
of this project.

Original Visualization in this project are a byproduct of training torchvision's
ResNet-18 variant with ArtBench, resulting in novel visualizations and findings
based on them.

Comparable results were achieved in terms of objectives, performance, while
simultaneously improving readability and reducing the total number of files
needed for operation, thus contributing to easier reproducibility by an amateur
user.

Visualizations are recreated/genereated through the use of an intuitively
designed gradio interface, reducing the amount an end user has to spend
code-side.
