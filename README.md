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
Note: If the Metal Performance Shaders backend is available, the model with train on the MacOS GPU. Otherwise it will train on the cpu
- You can run the absolute positional encoding model with the following command: `python3 absolute_pos_encoding_vit.py`
- You can run the absolute positional encoding model with the folliwing command: `python3 relative_pos_encoding_vit.py`

### Results:
- Abolute Positional Encoding:
    - MNIST: 0.820 Test accuracy
    - CIFAR10: 0.546 Test accuracy
- Subvariant 1: Same paramters across all heads:
    - MNIST:
        - General Learnable Function: 0.975 Test Accuracy
        - Monotonically Decreasing Function: 0.368 Test Accuracy
        - Ratio of Two Polynomial Functions: 0.978 Test Accuracy
    - CIFAR10:
        - General Learnable Function: 0.510, 0.516 Test Accuracy
        - Monotonically Decreasing Function: 0.210 Test Accuracy
        - Ratio of Two Polynomial Functions: 0.479 Test Accuracy
- Subvariant 2: Different Parameters for Different Heads
    - MNIST:
        - General Learnable Function: 0.973 Test Accuracy
        - Monotonically Decreasing Function: 0.199 Test Accuracy
        - Ratio of Two Polynomial Functions: 
    - CIFAR10:
        - General Learnable Function: 0.547
        - Monotonically Decreasing Function: 0.438
        - Ratio of Two Polynomial Functions: 
