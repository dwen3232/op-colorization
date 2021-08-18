# op-colorization
Colorizes One Piece manga pages using GANs. I plan on experimenting more with CycleGAN and ACL-GAN implementations, as the supervised pix2pix GAN requires too much time matching the raw and colored manga pages together. The current implementation I have completed uses the colored pages and the grayscale of the colored pages as the training pairs, which of course does not translate perfectly to the raw manga pages. Just removing miscellaneous images (indexes, author's notes, covers, etc.) from the colored pages dataset took upwards of 2 hours. Because the order and inclusion of these miscellaneous images is not consistent between the raw and colored pages, pruning them both for supervised training is unrealistic (more accurately, I would rather just implement an unsupervised model than deal with that). For copyright reasons, I will not include the datasets or the scripts I used to scrape said datasets in this repo.

Because the series is so long, the art style and environments change drastically over time. Here I will showcase some examples

![0009-013-gray](https://user-images.githubusercontent.com/55464175/129821695-474ede9c-690d-4a99-b145-063e35b04b42.png) | ![0009-013-color](https://user-images.githubusercontent.com/55464175/129821719-e44bef4b-7389-4b27-92d2-5a271a4b08fc.png) | ![0009-013-gen](https://user-images.githubusercontent.com/55464175/129821731-3ba10c44-3c3b-4f8f-aae7-5b0ccb4855af.png)

This page is pretty early in the series (chapter 9). The generator gets most of the colors correct (with some strange streaks of discoloration). This page is relatively simple, and the results are pretty good


