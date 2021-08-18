# op-colorization
This project aims to colorize One Piece manga pages using GANs. I plan on experimenting more with CycleGAN and ACL-GAN implementations, as the supervised pix2pix GAN requires too much time matching the raw and colored manga pages together. The current implementation I have completed uses the colored pages and the grayscale of the colored pages as the training pairs, which of course does not translate perfectly to the raw manga pages. Just removing miscellaneous images (indexes, author's notes, covers, etc.) from the colored pages dataset took upwards of 2 hours. Because the order and inclusion of these miscellaneous images is not consistent between the raw and colored pages, pruning them both for supervised training is unrealistic (more accurately, I would rather just implement an unsupervised model because I'm too lazy). For copyright reasons, I will not include the datasets or the scripts I used to scrape said datasets in this repo.

Because the series is so long, the art style and environments change drastically over time. Here I will showcase some examples. There may be some mild spoilers (nothing too important though), so if you care at all, turn away now.

![0009-013-gray](https://user-images.githubusercontent.com/55464175/129821695-474ede9c-690d-4a99-b145-063e35b04b42.png)  ![0009-013-color](https://user-images.githubusercontent.com/55464175/129821719-e44bef4b-7389-4b27-92d2-5a271a4b08fc.png)  ![0009-013-gen](https://user-images.githubusercontent.com/55464175/129821731-3ba10c44-3c3b-4f8f-aae7-5b0ccb4855af.png)

This page is pretty early in the series (chapter 9). The generator gets most of the colors correct (with some strange streaks of discoloration). This page is relatively simple, and the results are pretty good. The colors overall look a bit washed, however. Perhaps a larger lambda value would fix it?

![0238-014-gray](https://user-images.githubusercontent.com/55464175/129822010-51c017d2-9897-486a-ba30-27f15512dda2.png) ![0238-014-color](https://user-images.githubusercontent.com/55464175/129822030-f3dec794-8af6-44b7-b801-3e0e49ccb1df.png) ![0238-014-gen](https://user-images.githubusercontent.com/55464175/129822048-a6559c6e-ec8a-430d-bd38-dd5cd61295db.png)

Here, the gang is on a sky island (chapter 238 -- much further into the series). Suprisingly, the generator gets the color of the clouds correct, which is a very pleasant surprise. Like the previous example, the generated output colors are slightly grayer than they should be.

![0598-003-gray](https://user-images.githubusercontent.com/55464175/129822460-57afb8a4-b5db-46ce-bdf9-a64815755269.png) ![0598-003-color](https://user-images.githubusercontent.com/55464175/129822467-c16ccf30-e3b7-4335-a671-bab6353a5db2.png) ![0598-003-gen](https://user-images.githubusercontent.com/55464175/129822485-6e9c45bd-4fa7-4794-a806-62d59eb6caf4.png)

This image is the chapter cover for chapter 598. Chapter covers with this much detail are rare (maybe around 30ish in total?), so it's unsurprising that the model struggled with this one. The colors are mostly correct, but, like the last two, are slightly washed out.

![0604-003-gray](https://user-images.githubusercontent.com/55464175/129822656-b719e75d-c88d-48b5-a4b7-0422950e7d0f.png) ![0604-003-color](https://user-images.githubusercontent.com/55464175/129822661-2f7fcbbe-f0fa-4c7c-968c-3ff822e198c0.png) ![0604-003-gen](https://user-images.githubusercontent.com/55464175/129822666-44d92329-d4b0-4c46-9be4-ddcb73ee9dd0.png)

All of a sudden, in chapter 604, the gang is underwater! As expected, the generator struggled with this one. I have no clue what it thinks the water is, but it looks red for some reason.

![0909-008-gray](https://user-images.githubusercontent.com/55464175/129822788-0202de31-848b-4ba8-b5f2-59ddc6173f7f.png) ![0909-008-color](https://user-images.githubusercontent.com/55464175/129822797-cb9bdd73-4a41-48a8-8b3b-6aec755021c1.png) ![0909-008-gen](https://user-images.githubusercontent.com/55464175/129822800-dd2e11d6-965c-486c-a4f9-181854c491a2.png)

Just for fun, I wanted to see what the generated image would look like for one of the most intricate pages in the entire series all the way at chapter 909. The generated image is not nearly as colorful as the colored, but it still looks pretty cool! 

As I mentioned before, the supervised method does not translate well to the original pages at all. Here's what the generated output looks like using the raw page.

![0909-008-rawgen](https://user-images.githubusercontent.com/55464175/129823456-be9c1061-a97e-4757-beb0-1b10caeec514.png)

I surmise that the lack of shading is confusing the generator, as it relied on the implicit information given by the grayscale. Perhaps that's why the sky island example looked so good.


My next goal is to try training the model on more specific subsets of the manga. For my examples, I trained the model using all 900+ chapters that I scraped. The evolution of the style and the changing environments may be influencing the model in unexpected ways. I may also experiment with the lambda value; the value given by the original paper may not be the best suited for this dataset.

After that, I will try CycleGAN and ACL-GAN for raw page to color page conversion.

All the theoretical details relating to pix2pix GAN can be found in this paper: https://arxiv.org/abs/1611.07004
