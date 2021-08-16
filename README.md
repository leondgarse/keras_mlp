# Keras_mlp
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Keras_mlp](#kerasmlp)
	- [Usage](#usage)
	- [MLP mixer](#mlp-mixer)
	- [ResMLP](#resmlp)

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
  - [PDF 2105.01601 MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf).
  - [Github lucidrains/mlp-mixer-pytorch](https://github.com/lucidrains/mlp-mixer-pytorch).
  - Weights reload from [Github google-research/vision_transformer](https://github.com/google-research/vision_transformer#available-mixer-models).
  - **Models** `Top1 Acc` is `Pre-trained on JFT-300M` model accuray on `ImageNet 1K` from paper.
    | Model       | Params | Top1 Acc | ImageNet | Imagenet21k | ImageNet SAM |
    | ----------- | ------ | -------- | --------------- | ------------------ | ------------------- |
    | MlpMixerS32 | 19.1M  | 68.70    |                 |                    |                     |
    | MlpMixerS16 | 18.5M  | 73.83    |                 |                    |                     |
    | MlpMixerB32 | 60.3M  | 75.53    |                 |                    | [b32_imagenet_sam.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_b32_imagenet_sam.h5) |
    | MlpMixerB16 | 59.9M  | 80.00    | [b16_imagenet.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_b16_imagenet.h5) | [b16_imagenet21k.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_b16_imagenet21k.h5) | [b16_imagenet_sam.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_b16_imagenet_sam.h5) |
    | MlpMixerL32 | 206.9M | 80.67    |  |  |                     |
    | MlpMixerL16 | 208.2M | 84.82    | [l16_imagenet.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_l16_imagenet.h5) | [l16_imagenet21k.h5](https://github.com/leondgarse/keras_mlp/releases/download/mlp_mixer/mlp_mixer_l16_imagenet21k.h5) |                     |
    | - input 448 | 208.2M | 86.78    |                 |                    |                     |
    | MlpMixerH14 | 432.3M | 86.32    |                 |                    |                     |
    | - input 448 | 432.3M | 87.94    |                 |                    |                     |

    | Specification        | S/32  | S/16  | B/32  | B/16  | L/32  | L/16  | H/14  |
    | -------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
    | Number of layers     | 8     | 8     | 12    | 12    | 24    | 24    | 32    |
    | Patch resolution P×P | 32×32 | 16×16 | 32×32 | 16×16 | 32×32 | 16×16 | 14×14 |
    | Hidden size C        | 512   | 512   | 768   | 768   | 1024  | 1024  | 1280  |
    | Sequence length S    | 49    | 196   | 49    | 196   | 49    | 196   | 256   |
    | MLP dimension DC     | 2048  | 2048  | 3072  | 3072  | 4096  | 4096  | 5120  |
    | MLP dimension DS     | 256   | 256   | 384   | 384   | 512   | 512   | 640   |
  - **Usage** Parameter `pretrained` is added in value `[None, "imagenet", "imagenet21k", "imagenet_sam"]`. Default is `imagenet`.
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
    For `"imagenet21k"` pre-trained model, actual `num_classes` is `21843`.
  - **Exclude model top layers** by set `num_classes=0`.
  	```py
    import keras_mlp
    mm = keras_mlp.MlpMixerL16(num_classes=0, pretrained="imagenet")
    print(mm.output_shape)
    # (None, 196, 1024)

    mm.save('mlp_mixer_l16_imagenet-notop.h5')
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
## ResMLP
  - [PDF 2105.03404 ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf)
  - [Github facebookresearch/deit](https://github.com/facebookresearch/deit)
  - **Models** loaded `imagenet` weights are the `distilled` version from official.
    | Model      | Params | Image resolution | Top1 Acc | ImageNet |
    | ---------- | ------ | ---------------- | -------- | -------- |
    | ResMLP12   | 15M    | 224              | 77.8     | [resmlp12_imagenet.h5](https://github.com/leondgarse/keras_mlp/releases/download/resmlp/resmlp12_imagenet.h5) |             |
    | ResMLP24   | 30M    | 224              | 80.8     | [resmlp24_imagenet.h5](https://github.com/leondgarse/keras_mlp/releases/download/resmlp/resmlp24_imagenet.h5) |             |
    | ResMLP36   | 116M   | 224              | 81.1     | [resmlp36_imagenet.h5](https://github.com/leondgarse/keras_mlp/releases/download/resmlp/resmlp36_imagenet.h5) |             |
    | ResMLP_B24 | 129M   | 224              | 83.6     | [resmlp_b24_imagenet.h5](https://github.com/leondgarse/keras_mlp/releases/download/resmlp/resmlp_b24_imagenet.h5) |             |
    | - imagenet22k | 129M   | 224              | 84.4     | [resmlp_b24_imagenet22k.h5](https://github.com/leondgarse/keras_mlp/releases/download/resmlp/resmlp_b24_imagenet22k.h5) |             |

  - **Usage** Parameter `pretrained` is added in value `[None, "imagenet", "imagenet22k"]`, where `imagenet22k` means pre-trained on `imagenet21k` and fine-tuned on `imagenet`. Default is `imagenet`.
    ```py
    import keras_mlp
    # Will download and load `imagenet` pretrained weights.
    # Model weight is loaded with `by_name=True, skip_mismatch=True`.
    mm = keras_mlp.ResMLP24(num_classes=1000, pretrained="imagenet")

    # Run prediction
    from skimage.data import chelsea
    imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
    pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
    print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.86475366), ('n02123045', 'tabby', 0.028439553), ...]
    ```
  - **Exclude model top layers** by set `num_classes=0`.
  	```py
    import keras_mlp
    mm = keras_mlp.ResMLP_B24(num_classes=0, pretrained="imagenet22k")
    print(mm.output_shape)
    # (None, 784, 768)

    mm.save('resmlp_b24_imagenet22k-notop.h5')
  	```
## [In progress] GMLP
	- [PDF 2105.08050 Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050.pdf).
	- Model weights load from [Github timm/models/mlp_mixer](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mlp_mixer.py).
	- **Models**
    | Model   | Params | Image resolution | Top1 Acc | ImageNet |
    | ------- | ------ | ---------------- | -------- | -------- |
    | GMLP_Ti | 6M     | 224              | 72.3     |          |
    | GMLP_S  | 20M    | 224              | 79.6     |          |
    | GMLPB   | 73M    | 224              | 81.6     |          |
***
