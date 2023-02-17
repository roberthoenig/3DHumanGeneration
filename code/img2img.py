import replicate
img_path = "/home/robert/Downloads/image.png"
mask_path = "/home/robert/Downloads/image.png"
model = replicate.models.get("andreasjansson/stable-diffusion-inpainting")
version = model.versions.get("8eb2da8345bee796efcd925573f077e36ed5fb4ea3ba240ef70c23cf33f0d848")
output = version.predict(prompt="a sketch of a moving stick figure, very few lines, very few strokes, pencil sketch, minimal drawing", image=open(img_path, "rb"), mask=open(img_path, "rb"), invert_mask=True)
print("output", output)

"a pencil sketch of a human, quick drawing, like a comic, looks like a very thin stick figure"

# import replicate
# model = replicate.models.get("andreasjansson/stable-diffusion-inpainting")
# version = model.versions.get("8eb2da8345bee796efcd925573f077e36ed5fb4ea3ba240ef70c23cf33f0d848")
# output = version.predict(prompt="...")
