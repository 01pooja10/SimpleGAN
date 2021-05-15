# SimpleGAN
An introductory &amp; comprehensible re-implementation of the infamous Generative Adversarial Network (Goodfellow et al.) [paper](https://arxiv.org/abs/1406.2661), trained on MNIST images and written using the PyTorch framework.
![paper](data/gan_paper.jpg)
This is meant to be a beginner-friendly, working GAN model which can generate handwritten digits(1-10) and is visualized using TensorBoard. The results, upon successful completion of the project, will be displayed below. Various

## Implementation
The Generative Adversarial Network consists of 2 models namely, the generative and discriminative models which are Deep Neural Networks. The generative and discriminative models compete against each other in the same environment where, one (generative model) works to generate fake data and the other (discriminative model) tries to differentiate between the fake and original data.
