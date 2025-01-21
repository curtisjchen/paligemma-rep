# PaliGemma Reproduction

Coding a multimodal vision language model using PyTorch. Following Google's PaliGemma paper. 

The model uses a SigLip contrastive vision encoder and a linear projection layer into a Gemma, a pretrained LLM. This allows for tokenization and understanding of images in language space, which can be used for various tasks, such as VQA, segmentation, and object detection. 