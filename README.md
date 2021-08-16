# Keras_mlp
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Keras_mlp](#kerasmlp)
	- [Usage](#usage)
	- [MLP mixer](#mlp-mixer)
	- [[In progress] ResMLP](#in-progress-resmlp)

<!-- /TOC -->
***

## Usage
  - This repo can be installed as a pip package.
    ```sh
    pip install -U git+https://github.com/leondgarse/keras_mlp
    ```
    or just `git clone` it.
    ```sh
    git clone https://github.com/leondgarse/keras_mlp.git
    cd keras_mlp && pip install .
    ```
## MLP mixer
  - [PDF 2105.01601 MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf)
  - [Github lucidrains/mlp-mixer-pytorch](https://github.com/lucidrains/mlp-mixer-pytorch)
  - Weights reload from [Github google-research/vision_transformer](https://github.com/google-research/vision_transformer#available-mixer-models)
  - **Models**
    | Model       | Params | Top1 Acc | ImageNet | Imagenet21k | ImageNet SAM |
    | ----------- | ------ | -------- | --------------- | ------------------ | ------------------- |
    | MlpMixerS32 | 19.1M  |          |                 |                    |                     |
    | MlpMixerS16 | 18.5M  |          |                 |                    |                     |
    | MlpMixerB32 | 60.3M  |          |                 |                    | [b32_imagenet_sam.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_b32_imagenet_sam.h5) |
    | MlpMixerB16 | 59.9M  |          | [b16_imagenet.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_b16_imagenet.h5) | [b16_imagenet21k.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_b16_imagenet21k.h5) | [b16_imagenet_sam.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_b16_imagenet_sam.h5) |
    | MlpMixerL32 | 206.9M |          |  |  |                     |
    | MlpMixerL16 | 208.2M |          | [l16_imagenet.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_l16_imagenet.h5) | [l16_imagenet21k.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_l16_imagenet21k.h5) |                     |
    | MlpMixerH14 | 432.3M |          |                 |                    |                     |

    | Specification        | S/32  | S/16  | B/32  | B/16  | L/32  | L/16  | H/14  |
    | -------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
    | Number of layers     | 8     | 8     | 12    | 12    | 24    | 24    | 32    |
    | Patch resolution P×P | 32×32 | 16×16 | 32×32 | 16×16 | 32×32 | 16×16 | 14×14 |
    | Hidden size C        | 512   | 512   | 768   | 768   | 1024  | 1024  | 1280  |
    | Sequence length S    | 49    | 196   | 49    | 196   | 49    | 196   | 256   |
    | MLP dimension DC     | 2048  | 2048  | 3072  | 3072  | 4096  | 4096  | 5120  |
    | MLP dimension DS     | 256   | 256   | 384   | 384   | 512   | 512   | 640   |
  - **Usage** Parameter `pretrained` is added in value `[None, "imagenet", "imagenet21k", "imagenet_sam"]`, default is `imagenet`.
    ```py
    import keras_mlp
    # Will download and load `imagenet` pretrained weights.
    # Model weight is loaded with `by_name=True, skip_mismatch=True`.
    mm = keras_mlp.MlpMixerB16(num_classes=1000, pretrained="imagenet")

    # Run prediction
    from skimage.data import chelsea
    imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='tf') # Chelsea the cat
    pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
    print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.9568315), ('n02123045', 'tabby', 0.017994137), ...]
    ```
  - **Pre-training details**
    - We pre-train all models using Adam with β1 = 0.9, β2 = 0.999, and batch size 4 096, using weight decay, and gradient clipping at global norm 1.
    - We use a linear learning rate warmup of 10k steps and linear decay.
    - We pre-train all models at resolution 224.
    - For JFT-300M, we pre-process images by applying the cropping technique from Szegedy et al. [44] in addition to random horizontal flipping.
    - For ImageNet and ImageNet-21k, we employ additional data augmentation and regularization techniques.
    - In particular, we use RandAugment [12], mixup [56], dropout [42], and stochastic depth [19].
    - This set of techniques was inspired by the timm library [52] and Touvron et al. [46].
    - More details on these hyperparameters are provided in Supplementary B.
## [In progress] ResMLP
  - [PDF 2105.03404 ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf)
  - [Github rishikksh20/ResMLP-pytorch](https://github.com/rishikksh20/ResMLP-pytorch)
  - **Usage**
    ```py
    import keras_mlp
    model = keras_mlp.ResMLP12(num_classes=1000)
    ```
***
