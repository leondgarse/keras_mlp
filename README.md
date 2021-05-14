# Keras_mlp
***

## MLP mixer
  - [PDF MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf)
  - [Github lucidrains/mlp-mixer-pytorch](https://github.com/lucidrains/mlp-mixer-pytorch)
  - [Github Benjamin-Etheredge/mlp-mixer-keras](https://github.com/Benjamin-Etheredge/mlp-mixer-keras)
  - **Calculate total parameters** `DS -> token-mixing, DC --> channel-mixing, num_patch --> Sequence length`
    ```py
    inputs, patch_size, hidden_dim, DS, DC, num_blocks = 224, 32, 512, 256, 2048, 8
    num_patch = (inputs * inputs) // (patch_size * patch_size)
    conv = (patch_size * patch_size * 3 + 1) * hidden_dim
    mlp_1 = (num_patch + 1) * DS + (DS + 1) * num_patch # input_shape = (num_patch, hidden_dim)
    mlp_2 = (hidden_dim + 1) * DC + (DC + 1) * hidden_dim
    mixer = (2 * num_patch) * 2 + mlp_1 +  mlp_2  # mixer: (LN = 2 * num_patch) * 2 + mlp_1 +  mlp_2
    total = mixer * num_blocks + conv
    print(f'{total = }')  # total = 18575784
    ```

    | Specification        | S/32  | S/16  | B/32  | B/16  | L/32  | L/16  | H/14  |
    | -------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
    | Number of layers     | 8     | 8     | 12    | 12    | 24    | 24    | 32    |
    | Patch resolution P×P | 32×32 | 16×16 | 32×32 | 16×16 | 32×32 | 16×16 | 14×14 |
    | Hidden size C        | 512   | 512   | 768   | 768   | 1024  | 1024  | 1280  |
    | Sequence length S    | 49    | 196   | 49    | 196   | 49    | 196   | 256   |
    | MLP dimension DC     | 2048  | 2048  | 3072  | 3072  | 4096  | 4096  | 5120  |
    | MLP dimension DS     | 256   | 256   | 384   | 384   | 512   | 512   | 640   |
    | Parameters (M)       | 10    | 10    | 46    | 46    | 188   | 189   | 409   |
    | Self defined         | 18.5  | 18.0  | 59.5  | 59.1  | 205.8 | 207.1 | 430.9 |

  - **Pre-training details**
    - We pre-train all models using Adam with β1 = 0.9, β2 = 0.999, and batch size 4 096, using weight decay, and gradient clipping at global norm 1.
    - We use a linear learning rate warmup of 10k steps and linear decay.
    - We pre-train all models at resolution 224.
    - For JFT-300M, we pre-process images by applying the cropping technique from Szegedy et al. [44] in addition to random horizontal flipping.
    - For ImageNet and ImageNet-21k, we employ additional data augmentation and regularization techniques.
    - In particular, we use RandAugment [12], mixup [56], dropout [42], and stochastic depth [19].
    - This set of techniques was inspired by the timm library [52] and Touvron et al. [46].
    - More details on these hyperparameters are provided in Supplementary B.
***
