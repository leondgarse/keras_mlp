from keras_mlp.mlp_mixer import MlpMixer, MlpMixerS32, MlpMixerS16, MlpMixerB32, MlpMixerB16, MlpMixerL32, MlpMixerL16, MlpMixerH14
from keras_mlp.res_mlp import ResMLP, ResMLP12, ResMLP24, ResMLP36, ResMLP_B24
from keras_mlp.sam_model import SAMModel

__head_doc__ = """
Github source [leondgarse/keras_mlp](https://github.com/leondgarse/keras_mlp).
Keras implementation of [Github google-research/vision_transformer](https://github.com/google-research/vision_transformer#available-mixer-models).
Paper [PDF 2105.01601 MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
      For `"imagenet21k"` pre-trained model, actual `num_classes` is `21843`.
  activation: activation used in whole model, default `gelu`.
  sam_rho: None zero value to init model using `SAM` training step.
      SAM Arxiv article: [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf).
  dropout: dropout rate if top layers is included.
  drop_connect_rate:
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: value in [None, "imagenet", "imagenet21k", "imagenet_sam"].
      Will try to download and load pre-trained model weights if not None.
      Save path is `~/.keras/models/`.

Returns:
    A `keras.Model` instance.
"""

MlpMixer.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of layers.
  patch_size: stem patch resolution P×P, means `kernel_size=patch_size, strides=patch_size` for stem `Conv2D` block.
  hidden_dim: stem output channel dimenion.
  tokens_mlp_dim: MLP block token level hidden dimenion, where token level means `height * weight` dimention.
  channels_mlp_dim: MLP block channel level hidden dimenion.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model       | Params | Top1 Acc |
  | ----------- | ------ | -------- |
  | MlpMixerS32 | 19.1M  | 68.70    |
  | MlpMixerS16 | 18.5M  | 73.83    |
  | MlpMixerB32 | 60.3M  | 75.53    |
  | MlpMixerB16 | 59.9M  | 80.00    |
  | MlpMixerL32 | 206.9M | 80.67    |
  | MlpMixerL16 | 208.2M | 84.82    |
  | - input 448 | 208.2M | 86.78    |
  | MlpMixerH14 | 432.3M | 86.32    |
  | - input 448 | 432.3M | 87.94    |

  | Specification        | S/32  | S/16  | B/32  | B/16  | L/32  | L/16  | H/14  |
  | -------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
  | Number of layers     | 8     | 8     | 12    | 12    | 24    | 24    | 32    |
  | Patch resolution P×P | 32×32 | 16×16 | 32×32 | 16×16 | 32×32 | 16×16 | 14×14 |
  | Hidden size C        | 512   | 512   | 768   | 768   | 1024  | 1024  | 1280  |
  | Sequence length S    | 49    | 196   | 49    | 196   | 49    | 196   | 256   |
  | MLP dimension DC     | 2048  | 2048  | 3072  | 3072  | 4096  | 4096  | 5120  |
  | MLP dimension DS     | 256   | 256   | 384   | 384   | 512   | 512   | 640   |
"""


__default_doc__ = __head_doc__ + """
[Mixer architecture] num_blocks: {}, patch_size: {}, hidden_dim: {}, tokens_mlp_dim: {}, channels_mlp_dim: {}

Args:
""" + __tail_doc__

MlpMixerS32.__doc__ = __default_doc__.format(8, 32, 512, 256, 2048)
MlpMixerS16.__doc__ = __default_doc__.format(8, 16, 512, 256, 2048)
MlpMixerB32.__doc__ = __default_doc__.format(12, 32, 768, 384, 3072)
MlpMixerB16.__doc__ = __default_doc__.format(12, 16, 768, 384, 3072)
MlpMixerL32.__doc__ = __default_doc__.format(24, 32, 1024, 512, 4096)
MlpMixerL16.__doc__ = __default_doc__.format(24, 16, 1024, 512, 4096)
MlpMixerH14.__doc__ = __default_doc__.format(32, 14, 1280, 640, 5120)
