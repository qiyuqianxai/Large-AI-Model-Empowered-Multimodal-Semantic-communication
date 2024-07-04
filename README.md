# Large AI Model Empowered Multimodal Semantic Communications
## Authors
### Li Dong, Feibo Jiang, Yubo Peng, Kezhi Wang, Kun Yang, Cunhua Pan, Xiaohu You
## Paper
### https://arxiv.org/abs/2309.01249
## Code
### https://github.com/jiangfeibo/LAMMSC.git
## Abstract
Multimodal signals, including text, audio, image, and video, can be integrated into Semantic Communication (SC) system to provide an immersive experience with low latency and high quality at the semantic level. However, the multimodal SC has several challenges, including data heterogeneity, semantic ambiguity, and signal distortion during transmission. Recent advancements in large AI models, particularly in the Multimodal Language Model (MLM) and Large Language Model (LLM), offer potential solutions for addressing these issues. To this end, we propose a Large AI Model-based Multimodal SC (LAM-MSC) framework, where we first present the MLM-based Multimodal Alignment (MMA) that utilizes the MLM to enable the transformation between multimodal and unimodal data while preserving semantic consistency. Then, a personalized LLM-based Knowledge Base (LKB) is proposed, which allows users to perform personalized semantic extraction or recovery through the LLM. This effectively addresses the semantic ambiguity. Finally, we apply the Conditional Generative adversarial networks-based channel Estimation (CGE) for estimating the wireless channel state information. This approach effectively mitigates the impact of fading channels in SC. Finally, we conduct simulations that demonstrate the superior performance of the LAM-MSC framework.

![img](LAM-MSC.png)

## The function of each file
- [LAM-MSC.py](LAM-MSC.py): Overview of the LAM-MSC framework.

- [channel_nets.py](channel_nets.py): Definition of the channel encoder, channel decoder, and physical channel.

- [MMA.py](MMA.py): The implementation of the MMA module, including the modal transformation and recovery.

- [LKB.py](LKB.py): The implementation of the LKB module.

- [CGE.py](CGE.py): The implementation of the CGE module, including the network definition and training of CGAN.

- [SCwithCGE.py](SCwithCGE.py): The implementation of the image SC and CGE modules, including the training of the SC model.

- [CoDi](CoDi): The implementation of CoDi. The details refer to https://github.com/microsoft/i-Code/.

- [logs](logs): Path to save the logs during training.

- [checkpoints](checkpoints): Path to save model weights.

## Citation   
```
@article{jiang2023large,
  title={Large AI model empowered multimodal semantic communications},
  author={Jiang, Feibo and Peng, Yubo and Dong, Li and Wang, Kezhi and Yang, Kun and Pan, Cunhua and You, Xiaohu},
  journal={arXiv preprint arXiv:2309.01249},
  year={2023}
}
```

