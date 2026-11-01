[ ] 
[Machine Learning Research](https://machinelearning.apple.com/)
[Open Menu](https://machinelearning.apple.com/research/on-device-scene-analysis#localnav-menustate)
[Close Menu](https://machinelearning.apple.com/research/on-device-scene-analysis#)
- [Overview](https://machinelearning.apple.com/)
- [Research Highlights](https://machinelearning.apple.com/highlights)
- [Publications](https://machinelearning.apple.com/research/)
- [Events](https://machinelearning.apple.com/updates/)
- [Work with us](https://machinelearning.apple.com/work-with-us)
[content type highlight](https://machinelearning.apple.com/highlights)published June 7, 2022
[research area Computer Vision](https://machinelearning.apple.com/highlights?domain=Computer%20Vision), [research area Methods and Algorithms](https://machinelearning.apple.com/highlights?domain=Methods%20and%20Algorithms)

# 
A Multi\-Task Neural Architecture for On\-Device Scene Analysis


Scene analysis is an integral core technology that powers many features and experiences in the Apple ecosystem. From visual content search to powerful memories marking special occasions in one’s life, outputs \(or “signals”\) produced by scene analysis are critical to how users interface with the photos on their devices. Deploying dedicated models for each of these individual features is inefficient as many of these models can benefit from sharing resources. We present how we developed Apple Neural Scene Analyzer \(ANSA\), a unified backbone to build and maintain scene analysis workflows in production. This was an important step towards enabling Apple to be among the first in the industry to deploy fully client\-side scene analysis in 2016.

Although most extensively used in the Photos app, other noteworthy adopters of ANSA include:
- Camera, Spotlight search, and Accessibility features 
- Apple apps including Notes and Final Cut Pro 
- Apps from developers that use Apple’s [Vision APIs](https://developer.apple.com/documentation/vision) 

Our newest operating systems \(iOS 16 and Mac OS Ventura\) employ a single backbone that supports multiple tasks completely on\-device, and can execute all tasks in tens of milliseconds. Our main priority was to have a state\-of\-the\-art algorithmic workflow delivering world\-class user experiences while running all compute fully on the client. Running all compute on a user’s device means ANSA has to operate under tight power and performance constraints, while ensuring various user experiences receive high\-quality signals. This allowed us to be discerning with our model’s capacity while meeting user needs. Here are some important factors that influenced our deployment on the device:
1. Since devices have limited disk space, deploying multiple tasks was possible only if the backbone was shared and amortized compute and network parameters. 
2. Scene analysis has to be responsive for interactive use cases, so our latency targets had to be under tens of milliseconds. 
3. The neural architectures we consider have to be supported on heterogeneous deployment targets. All supported iPhone, iPad, Mac, and Apple TV models have slightly different hardware and compute capabilities, but all must execute ANSA locally. Therefore, we needed to design ANSA in such a way that it could execute efficiently across all applicable device hardware configurations. Although Transformer\-based vision backbones are promising, we wanted to offer these capabilities on devices that do not have a Neural Engine. Given this limitation, we chose to utilize architectures built with convolutional layers. 

In this article we share on how we train and deploy the network backbone. We begin by providing an overview of the algorithmic considerations as it relates to our legacy and new backbones. Next, we discuss the multihead network performance across various deployment targets, and discuss model quality. Finally, we provide additional context around the experiences powered by these signals.
[![Figure 1: Visual Backbone.](https://mlr.cdn-apple.com/media/Fig1_visual_backbone_859a68eb37.png)](https://mlr.cdn-apple.com/media/Fig1_visual_backbone_859a68eb37.png)
Figure 1: The visual backbone supports several downstream tasks. Sharing the backbone across a variety of tasks allows for a compact footprint on the device. The portion of the network shown in orange is the image portion of the image\-language backbone trained with a contrastive objective. Note that the language portion of the backbone is not shown in the illustration above. Tasks branching lower in the orange tower rely on features closer to the input image, while those branching near the top rely on higher level features closer to the final embedding layer. Tasks that branch at a similar depth share colors in this figure.

## [Supporting Multiple Downstream Tasks with a Systemwide Image\-Language Backbone](https://machinelearning.apple.com/research/on-device-scene-analysis#supporting-multiple-downstream-tasks-with-a-systemwide-image-language-backbone)

Our goal is to train a high\-quality image backbone that can power a large variety of user experiences. Prior to iOS and macOS Ventura, ANSA was designed around a frozen backbone trained with a multilabel classification objective. This choice was intended to yield a robust and universal representation of images for our additional downstream tasks, which were trained separately.

Initially, this classifier was trained on a few million images that included stock imagery. However, recent work has shown that leveraging [large\-scale weakly supervised data sets](https://arxiv.org/abs/2102.05918) can enable [state\-of\-the\-art performance](https://arxiv.org/abs/2103.00020) on many downstream tasks. These models are trained to jointly embed images with their corresponding natural\-language descriptions using a [contrastive objective](https://arxiv.org/abs/1807.03748) without requiring any [high\-quality task\-specific annotations](https://arxiv.org/abs/2002.05709). Furthermore, these image\-language models learn useful representations that can be used for transfer learning on a variety of downstream tasks, while having the ability to perform zero\-shot transfer for image tagging in a way that is competitive with fully supervised models. Our newer backbone adopts the joint image\-language paradigm and is trained on a few hundred million image\-text pairs.

Using large\-scale datasets required parallel training on a large number of GPUs \(typically hundreds\), while employing techniques such as [gradient checkpointing](https://arxiv.org/abs/1604.06174) and [mixed\-precision training](https://arxiv.org/abs/1710.03740) to maximize the local batch size on each GPU. Furthermore, to ensure the input pipeline doesn’t become a bottleneck, we shard the data set across GPU workers and stored the data set with images resized to a resolution closer to what the model needs for training, enabling faster training throughput \(~10x the speed of a base implementation\).

With this new image\-language pretrained backbone, we transfer learning for other tasks using subnetworks that branch off at various layers in the frozen backbone, as shown in [Figure 1](https://machinelearning.apple.com/research/on-device-scene-analysis#figure1). Previously classification was used as the main pre\-training objective for the backbone itself, the classification task is now also supported by a small head deployed on top of the image\-language backbone. By switching to a [MobileNetv3](https://arxiv.org/abs/1905.02244) backbone trained with an image\-language objective, we see a 10.5 percent improvement in mean average precision compared to directly using the original classification pretrained backbone. The heads used can vary from a simple linear layer on top of the pretrained image embedding \(like the one used for the utility content filter\) to a more complex convolutional network connected to different layers in the backbone \(like the one used for the object detection\). The [object detection head](https://arxiv.org/abs/1904.07850) adds only 1 MB to the backbone’s disk usage.

## [On\-Device Deployment: Performance Characteristics on the Apple Neural Engine](https://machinelearning.apple.com/research/on-device-scene-analysis#on-device-deployment-performance-characteristics-on-the-apple-neural-engine)

A critical requirement of ANSA’s design was that all compute takes place on the user’s device, with no involvement of cloud or remote resources. This stipulation imposed several important constraints on the design of ANSA. Namely, the inference must be fast, require a limited amount of memory, use minimal amounts of power, and have as small a footprint on the disk as possible. After a thorough architecture search, we settled on a variant of [MobileNetv3](https://arxiv.org/abs/1905.02244), with modifications to better suit the A11 Bionic, M1, and later chips that accelerate machine learning tasks. In its final design, ANSA contains 16 million parameters, is able to execute all of its heads in under 9.7 milliseconds, using only around 24.6 MB of memory on the Apple Neural Engine \(ANE\), while occupying only around 16.4 MB of disk space. This was achieved through a variety of optimizations, including quantization and pruning techniques.

The text encoder contains 44 million parameters, and is a fairly large model to deploy on the client. We employ a number of techniques to reduce both the model size on disk and the runtime latency and peak memory usage. We employ [pruning techniques](https://arxiv.org/abs/1710.01878) while fine\-tuning the text encoder to remove about 40 percent of model weights without impacting evaluation metrics. Furthermore, we quantize the model weights to 8\-bit unsigned integers, which reduces the model size by another factor of four. Utilizing a pruned model allows for faster execution of the model \(3\-4 ms\) on the ANE, and results in a more compact representation of the model on disk.

### Performance Analysis
Multi\-headed ANSAText and Image Encoders
Figure 2: Performance Analysis of the models. The Multiheaded ANSA tab compares the performance of the iOS 15 model \(trained with a classification objective\), and iOS 16 \(trained with an image\-language objective\). Note that ANSA iOS 16 includes an additional perceptual print head that was not part of the older model but is still considerably more efficient. Across all performance metrics, the iOS 16 model does significantly better while operating on larger input resolutions. The Text and Image Encoders tab compares relative latency / memory / storage requirements between the visual and text encoders in iOS 16. The text encoder has only 26M parameters after pruning. Note that the visual encoder model capacity we had was very limited, since we did not want any existing user experiences to be impacted by performance regressions.

In [Figure 2](https://machinelearning.apple.com/research/on-device-scene-analysis#figure2), we illustrate performance characteristics of the current model in iOS&nbsp16; \(image\-language pretrained\), and contrast it with its predecessor in iOS 15 \(image classification pretrained\). On iPhone 13, we can run the backbone and all its tasks in just under 9.7 milliseconds while utilizing a peak memory of under 24.6 MB. Similarly, in comparison to its predecessor in iOS 15, the new model is significantly faster on the ANE, while demonstrating significant gains on older GPU devices. It also has an impressive 40 percent reduction in peak memory used during inference while requiring less storage on disk. [Figure 2](https://machinelearning.apple.com/research/on-device-scene-analysis#figure2) captures the latency and memory profile of the image and text encoders separately.

## [Considerations for Evaluating Model Quality](https://machinelearning.apple.com/research/on-device-scene-analysis#considerations-for-evaluating-model-quality)

We evaluate ANSA through a variety of targeted benchmarks for both the backbone and each downstream task. For the backbone, we evaluate the quality of the joint image\-text representation through retrieval metrics using held\-out image\-text data sets. Additionally, we conduct evaluations based around linear probing and zero\-shot accuracy of the pretrained backbone on public benchmarks including [ImageNet](https://ieeexplore.ieee.org/document/5206848) and [COCO](https://arxiv.org/abs/1405.0312). Table 1 shows model quality on relevant ImageNet and Coco benchmarks. We compare ANSA against our internal implementation of a ViT\-B\-16 based visual encoder that achieves quality that is on par with published [state\-of\-the\-art](https://arxiv.org/abs/2103.00020). The compact model leads to a quality hit on public benchmarks, but we found that the visual encoder was meeting task\-specific production KPIs.

| Vision architecture | \# Parameters \(vision\) | ImageNet Top\-5 Accuracy | Text to Image Coco Top\-5 Retrieval Accuracy |
|---|---|---|---|
| ViT\-B/16 | 86M | 87.54 | 64.94 |
| MobilenetV3  | 9M | 80.44 | 57.32 |

For each downstream task, we also have task\-specific measurements, such as mean average precision \(mAP\) for our classification task. These are typically performed on a mixture of held\-out data and user\-representative libraries that are intended to mimic the expected distribution of a typical user’s library. In addition to tracking metrics on our dedicated test sets for each task, we leverage Apple internal testing programs, gathering valuable feedback to complement our analysis with an even larger, evolving, and diverse population.

Prior to shipping, we benchmark our models along different dimensions relevant to the final user experience, especially concerning bias and fairness. We address any soft spots through targeted data collections covering the diverse appearance of subjects, poor lighting, images from older sensors, and changes in resolution, to cite a few. These data sets are built by mining large\-scale unlabeled repositories using embedding\-based image search, natural language retrieval, and model distance to decision\-boundaries. We incorporate this additional data after it has been reviewed carefully by a diverse set of annotators. The annotators correct the identified issues and prevent future regressions in our training and validation sets.

## [A Few Key User Experiences Leveraging Scene Analysis](https://machinelearning.apple.com/research/on-device-scene-analysis#a-few-key-user-experiences-leveraging-scene-analysis)

The signals generated by ANSA are used in a multitude of experiences across the system including visual search, Featured Photos, [Memories](https://support.apple.com/en-us/HT207023) that appear in the Photos app and Widget, wallpaper suggestions, deduplication, and scene classification in Camera.

Visual content search is an important use case that is serviced through a fixed taxonomy image tagger. Synonyms of tags in the taxonomy are searchable, while a word\-embedding model is used to mitigate null queries suggesting the closest searchable concept. These experiences are accessible through the Photos app and Spotlight. It is worth noting that Spotlight enables search across multiple apps. An example of the entry point into the Photos Search experience is shown in [Figure 3](https://machinelearning.apple.com/research/on-device-scene-analysis#figure3).

Memories in the Photos app is an experience that requires detailed visual content understanding. The scene tags from ANSA are ingested in the Photos knowledge graph, which is the foundation of personalized experiences across the user’s photo library. Detecting food, mountains, beaches, birthdays, pets, or hikes helps build a personalized understanding of the user and their interests. This in turn allows for an experience that offers personalized and delightful curated content. Other uses of ANSA include ranking imagery for cover photo selection, titles for generated video memories, and alignment with music content that enhances the experience. The use of image\-language embeddings has allowed the discovery of richer memory types that were previously inaccessible through fixed taxonomy tagging. Detecting human\-object interactions like people seated around a table is one such use case. As an example, the top left illustration in [Figure 3](https://machinelearning.apple.com/research/on-device-scene-analysis#figure3) leverages scene\-level tags such as beaches, mountains, and sunsets to generate a personalized Memories movie presentation.

Certain images, containing receipts, documents, and repair references may not be a suited for a variety of Photos features, including Memories, Sharing, and Featured Photos. ANSA has dedicated classifiers to prevent this from happening. The embedding generated by ANSA is used for image deduplication, custom classifiers for utility content filtering, and detection of low\-quality image content. Other interesting experiences include intelligent auto\-crop for UI placement, image ranking, and training personalized models for recommendation. The image on the bottom right of [Figure 3](https://machinelearning.apple.com/research/on-device-scene-analysis#figure3) illustrates Photos curation, which utilizes among other signals, utility content filtering, image tagging and image aesthetics to pick the highest\-quality images from the user’s library.

### Signals and Impacted Experiences
Figure 3. A few system experiences that leverage predictions from scene analysis, ranging from Photos Search, Memories, Photos Curation, Photo editing styles, and Personalized suggestions.

## [Conclusion](https://machinelearning.apple.com/research/on-device-scene-analysis#conclusion)

This article provides a closer look into our production scene\-analysis workflows. We emphasize deploying neural networks on the user’s device for local and privacy\-preserving compute. Our workflows have scaled and have kept pace with advancements in machine learning research over the past six years, overcoming the challenge of running efficiently our networks on the client side while maintaining state\-of\-the\-art quality for our users. This is key to a variety of experiences in the Apple ecosystem that seamlessly utilize scene analysis.

The work shown in this article is a result of years of collaboration between teams across Apple. To ensure ANSA would meaningfully enhance user experiences across the Apple ecosystem, these teams collaborated closely on data collection, modeling, training / deployment infrastructure, model quality and robustness in the field, and the final user experience. This article illustrated some of the unique opportunities for ML research and development in a highly integrated environment, covering aspects ranging from hardware, software engineering, and design. All this is to enable experiences that will positively impact the lives of millions of users.

## [Acknowledgments](https://machinelearning.apple.com/research/on-device-scene-analysis#acknowledgments)

Many people contributed to this research, including Brandon McKinzie, Rohan Chandra, Sachin Ravi, Alp Aygar, Anindya Sarkar, Atila Orhon, Côme Weber, Mandakinee Singh Patel, Sowmya Gopalan, and Vignesh Jagadeesh.

## [References](https://machinelearning.apple.com/research/on-device-scene-analysis#references)

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. “Training Deep Nets with Sublinear Memory Cost.” April 2016. [\[link.\]](https://arxiv.org/abs/1604.06174)

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. “A Simple Framework for Contrastive Learning of Visual Representations.” July 2020. [\[link.\]](https://arxiv.org/abs/2002.05709)

Jia Deng, Wei Dong, Richard Socher, Li\-Jia Li, Kai Li, and Li Fei\-Fei. “ImageNet: A Large\-Scale Hierarchical Image Database.” 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248–255. [\[link.\]](https://ieeexplore.ieee.org/document/5206848)

Andrew Howard, Mark Sandler, Grace Chu, Liang\-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and Hartwig Adam. “Searching for MobileNetV3.” November 2019. [\[link.\]](https://arxiv.org/abs/1905.02244)

Chao Jia, Yinfei Yang, Ye Xia, Yi\-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. “Scaling Up Visual and Vision\-Language Representation Learning with Noisy Text Supervision.” June 2021. [\[link.\]](https://arxiv.org/abs/2102.05918)

Tsung\-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. “Microsoft COCO: Common Objects in Context.” February 2015. [\[link.\]](https://arxiv.org/abs/1405.0312)

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. “Mixed Precision Training.” arXiv:1710.03740 \(2017\). [\[link.\]](https://arxiv.org/abs/1710.03740)

Aaron van den Oord, Yazhe Li, and Orial Vinyals. “Representation Learning with Contrastive Predictive Coding.”arXiv:1807.03748 \(2018\). [\[link.\]](https://arxiv.org/abs/1807.03748)

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. “Learning Transferable Visual Models from Natural Language Supervision.” February 2021. [\[link.\]](https://arxiv.org/abs/2103.00020)

Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. “LAION\-400M: Open Dataset of CLIP\-Filtered 400 Million Image\-Text Pairs.” November 2021. [\[link.\]](https://arxiv.org/abs/2111.02114)

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. “Conceptual Captions: A Cleaned, Hypernymed, Image Alt\-Text Dataset for Automatic Image Captioning.” Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics \(Long Papers\), Melbourne, Australia, July 15–20, 2018, 2556–256. [\[link.\]](https://aclanthology.org/P18-1238.pdf)

Xingyi Zhou, Dequan Wang, and Philipp Krähenbühl. “Objects as Points.” April 2019. [\[link.\]](https://arxiv.org/abs/1904.07850)

Michael Zhu, and Suyog Gupta. “To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression.” November 2017. [\[link.\]](https://arxiv.org/abs/1710.01878)

Tsung\-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. “Microsoft COCO: Common Objects in Context.” February 2015. [\[link.\]](https://link.springer.com/content/pdf/10.1007/978-3-319-10602-1_48.pdf)

## Related readings and updates.

### 
On\-device Panoptic Segmentation for Camera Using Transformers
[https://machinelearning.apple.com/research/panoptic-segmentation](https://machinelearning.apple.com/research/panoptic-segmentation)

October 19, 2021[research area Computer Vision](https://machinelearning.apple.com/highlights?domain=Computer%20Vision), [research area Methods and Algorithms](https://machinelearning.apple.com/highlights?domain=Methods%20and%20Algorithms)

Camera \(in iOS and iPadOS\) relies on a wide range of scene\-understanding technologies to develop images. In particular, pixel\-level understanding of image content, also known as *image segmentation*, is behind many of the app’s front\-and\-center features. Person segmentation and depth estimation powers Portrait Mode, which simulates effects like the shallow depth of field and Stage Light. Person and skin segmentation power semantic rendering in group shots of up to four people, optimizing contrast, lighting, and even skin tones for each subject individually. Person, skin, and sky segmentation power Photographic Styles, which creates a personal look for your photos by selectively applying adjustments to the right areas guided by segmentation masks, while preserving skin tones. Sky segmentation and skin segmentation power denoising and sharpening algorithms for better image quality in low\-texture regions. Several other features consume image segmentation as an essential input.
[Read more](https://machinelearning.apple.com/research/panoptic-segmentation)

### 
Recognizing People in Photos Through Private On\-Device Machine Learning
[https://machinelearning.apple.com/research/recognizing-people-photos](https://machinelearning.apple.com/research/recognizing-people-photos)

July 28, 2021[research area Computer Vision](https://machinelearning.apple.com/highlights?domain=Computer%20Vision), [research area Privacy](https://machinelearning.apple.com/highlights?domain=Privacy)

Photos \(on iOS, iPad OS, and Mac OS\) is an integral way for people to browse, search, and relive life’s moments with their friends and family. Photos uses a number of machine learning algorithms, running privately on\-device, to help curate and organize images, Live Photos, and videos. An algorithm foundational to this goal recognizes people from their visual appearance.
[Read more](https://machinelearning.apple.com/research/recognizing-people-photos)
![Bottom banner](https://mlr.cdn-apple.com/media/Discover_1440x420_2x_9c465d585e.jpg)

## Discover opportunities in Machine Learning.

Our research in machine learning breaks new ground every day.

[Work with us](https://machinelearning.apple.com/work-with-us)
1. [Machine Learning Research](https://machinelearning.apple.com/)
2. [Publications](https://machinelearning.apple.com/research)
3. 

A Multi\-Task Neural Architecture for On\-Device Scene Analysis
- [Privacy Policy](https://www.apple.com/legal/privacy/)
- [Terms of Use](https://www.apple.com/legal/internet-services/terms/site.html)
- [Legal](https://www.apple.com/legal/)
Copyright © 2026 [Apple Inc.](https://www.apple.com/) All rights reserved.
 

 
