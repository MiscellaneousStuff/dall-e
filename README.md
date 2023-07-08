# DALL-E

## About

Re-implementation of Dall-E.

## Method

- [x] Use VQ-VAE to encode and decode images.
- [ ] Ingest text tokens and predict VQ-VAE Codes
   - [x] Use megabyte model (Will also allow massive context length)
   - [ ] Just encode text using chars for now
   - [ ] Auto-regressively predict VQ-VAE codes from text tokens