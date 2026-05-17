1. Start with the math/extraction layer (no ML dependencies, easiest to test in isolation):

skin_profile.py — the ITA + hue angle extractor
mask_generator.py — the MediaPipe region masks

2. Then the data layer:

dataset.py — depends on the two above, so they need to work first

3. Then the model files (order matters because train.py imports all of them):

unet.py
diffusion.py
context_encoder.py
4. Last:

train.py — imports everything above
demo/inference script — runs after training is done