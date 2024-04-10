# Examining Various Relative Positional Encoding (RPE) Mechanisms for Vision Transformers (ViTs)

### RPE Mechanisms Explored
- General Learnable Function: $f_\Theta : \mathbb{R} \rightarrow \mathbb{R}$
- Monotonically Decreasing Function: $f = e^{-\alpha x}$
- Ratio of two polynomial functions: $f = \frac{h}{g}$

Each mechanism is tested on MNIST and CIFAR10