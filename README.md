# DALL-E

## About

Re-implementation of Dall-E.

## Model

This repository is loosely based on the original DALL-E paper by OpenAI.
Instead of using a GPT2/GPT3 like autoregressive transformer decoder
architecture, it uses the [Megabyte](https://github.com/MiscellaneousStuff/megabyte)
based model from lucidrains.

## Method

- [x] Use VQ-VAE to encode and decode images.
- [x] Ingest text tokens and predict VQ-VAE Codes
   - [x] Use megabyte model (Will also allow massive context length)
   - [x] Just encode text using chars for now
   - [x] Auto-regressively predict VQ-VAE codes from text tokens
- [x] CIFAR 10 results bad. Perhaps because VQ-VAE bad with images below 64x64, switching to Tiny ImageNet. (NOTE: There was issues processing data, nothing to do with CIFAR-10).
- Tiny ImageNet
   - [x] Validate Tiny ImageNet captions and images (so they matchup)
      - Labels needed to be sorted same as the trainloader/testloader.
   - [x] Overfit DALL-E model on one caption image pair.
   - [x] Overfit DALL-E model on one batch of caption image pairs.
- Switching back to CIFAR-10 as there are far more examples per class vs Tiny ImageNet.
   - [ ] Overfit DALL-E model on one caption image pair.
   - [ ] Overfit DALL-E model on one batch of caption image pairs.