<hr>
<h1 align="center">
  DenoMamba <br>
  <sub>A fused state-space model for low-dose CT denoising</sub>
</h1>

<div align="center">
  <a href="https://avesis.hacibayram.edu.tr/saban.ozturk" target="_blank">Şaban&nbsp;Öztürk</a>;
  <a href="https://www.linkedin.com/in/oguz-can-duran/" target="_blank">Oğuz&nbspCan Duran</a>;
  <a href="https://kilyos.ee.bilkent.edu.tr/~cukur/" target="_blank">Tolga&nbsp;Çukur</a>;
  
<hr>

<h3 align="center">[<a href="https://arxiv.org/abs/2405.14022">arXiv</a>]</h3>

Official PyTorch implementation of **DenoMamba**, a novel denoising method based on state-space modeling (SSM), that efficiently captures short- and long-range context in medical images. Following an hourglass architecture with encoder-decoder stages, DenoMamba employs a spatial SSM module to encode spatial context and a novel channel SSM module equipped with a secondary gated convolution network to encode latent features of channel context at each stage. Feature maps from the two modules are then consolidated with low-level input features via a convolution fusion module (CFM).


![architecture](figures/architecture.png)



Copyright © 2024, ICON Lab.
