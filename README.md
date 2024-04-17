# Examining Various Relative Positional Encoding (RPE) Mechanisms for Vision Transformers (ViTs)

### RPE Mechanisms Explored
- General Learnable Function: $f_\Theta : \mathbb{R} \rightarrow \mathbb{R}$
- Monotonically Decreasing Function: $f = e^{-\alpha x}$
- Ratio of two polynomial functions: $f = \frac{h}{g}$

### Subvariants
We explore two subvariants for each of the mechanisms above:
- Each attention head of a layer uses the same parameters of the corresponding RPE mechanism
- Separate heads use separate parameters of the corresponding RPE mechamism

We also provide a comparison to the RPE-free baseline, which can be seen in `absolute_pos_encoding_vit.py`

Each mechanism is tested on MNIST, CIFAR datasets

### How to Run:
- You can run the absolute positional encoding model with the following command: `python3 absolute_pos_encoding_vit.py`
    - If "mps" is available the model with train on the MacOS GPU. Otherwise it will train on the cpu

### Results:
- Absolute Positional Encoding: 0.820 Test accuracy
- General Learnable Function: 0.975, 0.973  Test accuracy
- Monotonically Decreasing Function: 
- Ratio of Two Polynomial Functions: 