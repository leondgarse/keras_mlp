from keras_mlp.mlp_mixer import MlpMixer, MlpMixerS32, MlpMixerS16, MlpMixerB32, MlpMixerB16, MlpMixerL32, MlpMixerL16, MlpMixerH14
from keras_mlp.res_mlp import ResMLP, ResMLP12, ResMLP24, ResMLP36, ResMLP_B24
from keras_mlp.sam_model import SAMModel

__mlp_mixer_head_doc__ = """
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
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the conv branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: value in {pretrained_list}.
      Will try to download and load pre-trained model weights if not None.
      Save path is `~/.keras/models/`.

Returns:
    A `keras.Model` instance.
"""

MlpMixer.__doc__ = __mlp_mixer_head_doc__ + """
Args:
  num_blocks: number of layers.
  patch_size: stem patch resolution P×P, means `kernel_size=patch_size, strides=patch_size` for stem `Conv2D` block.
  hidden_dim: stem output channel dimenion.
  tokens_mlp_dim: MLP block token level hidden dimenion, where token level means `height * weight` dimention.
  channels_mlp_dim: MLP block channel level hidden dimenion.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "imagenet21k", "imagenet_sam"]) + """
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


__mixer_default_doc__ = __mlp_mixer_head_doc__ + """
[{model_name} architecture] num_blocks: {num_blocks}, patch_size: {patch_size}, hidden_dim: {hidden_dim}, tokens_mlp_dim: {tokens_mlp_dim}, channels_mlp_dim: {channels_mlp_dim}.

Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "imagenet21k", "imagenet_sam"])

MlpMixerS32.__doc__ = __mixer_default_doc__.format(model_name="MlpMixerS32", **mlp_mixer.BLOCK_CONFIGS["s32"])
MlpMixerS16.__doc__ = __mixer_default_doc__.format(model_name="MlpMixerS16", **mlp_mixer.BLOCK_CONFIGS["s16"])
MlpMixerB32.__doc__ = __mixer_default_doc__.format(model_name="MlpMixerB32", **mlp_mixer.BLOCK_CONFIGS["b32"])
MlpMixerB16.__doc__ = __mixer_default_doc__.format(model_name="MlpMixerB16", **mlp_mixer.BLOCK_CONFIGS["b16"])
MlpMixerL32.__doc__ = __mixer_default_doc__.format(model_name="MlpMixerL32", **mlp_mixer.BLOCK_CONFIGS["l32"])
MlpMixerL16.__doc__ = __mixer_default_doc__.format(model_name="MlpMixerL16", **mlp_mixer.BLOCK_CONFIGS["l16"])
MlpMixerH14.__doc__ = __mixer_default_doc__.format(model_name="MlpMixerH14", **mlp_mixer.BLOCK_CONFIGS["h14"])

__resmlp_head_doc__ = """
Github source [leondgarse/keras_mlp](https://github.com/leondgarse/keras_mlp).
Keras implementation of [Github facebookresearch/deit](https://github.com/facebookresearch/deit).
Paper [PDF 2105.03404 ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf).
"""

ResMLP.__doc__ = __resmlp_head_doc__ + """
Args:
  num_blocks: number of layers.
  patch_size: stem patch resolution P×P, means `kernel_size=patch_size, strides=patch_size` for stem `Conv2D` block.
  hidden_dim: stem output channel dimenion.
  channels_mlp_dim: MLP block channel level hidden dimenion.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "imagenet22k"]) + """
Model architectures:
  | Model         | Params | Image resolution | Top1 Acc |
  | ------------- | ------ | ---------------- | -------- |
  | ResMLP12      | 15M    | 224              | 77.8     |
  | ResMLP24      | 30M    | 224              | 80.8     |
  | ResMLP36      | 116M   | 224              | 81.1     |
  | ResMLP_B24    | 129M   | 224              | 83.6     |
  | - imagenet22k | 129M   | 224              | 84.4     |
"""

__resmlp_default_doc__ = __resmlp_head_doc__ + """
[{model_name} architecture] num_blocks: {num_blocks}, patch_size: {patch_size}, hidden_dim: {hidden_dim}, channels_mlp_dim: {channels_mlp_dim}.

Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "imagenet22k"])

ResMLP12.__doc__ = __resmlp_default_doc__.format(model_name="ResMLP12", **res_mlp.BLOCK_CONFIGS["12"])
ResMLP24.__doc__ = __resmlp_default_doc__.format(model_name="ResMLP24", **res_mlp.BLOCK_CONFIGS["24"])
ResMLP36.__doc__ = __resmlp_default_doc__.format(model_name="ResMLP36", **res_mlp.BLOCK_CONFIGS["36"])
ResMLP_B24.__doc__ = __resmlp_default_doc__.format(model_name="ResMLP_B24", **res_mlp.BLOCK_CONFIGS["b24"])
