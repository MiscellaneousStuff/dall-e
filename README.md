# DALL-E

## About

Re-implementation of Dall-E.

## Method

- [x] Use VQ-VAE to encode and decode images.
- [x] Ingest text tokens and predict VQ-VAE Codes
   - [x] Use megabyte model (Will also allow massive context length)
   - [x] Just encode text using chars for now
   - [x] Auto-regressively predict VQ-VAE codes from text tokens
- [x] CIFAR 10 results bad. Perhaps because VQ-VAE bad with images below 64x64, switching to Tiny ImageNet.
- [ ] Validate Tiny ImageNet captions and images (so they matchup)