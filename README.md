# ComfyUi-MiniMax-H3-Image-And-Reference-To-Video

This custom node lets you use I2V and reference images on MiniMax-H3 simultaneously.

https://github.com/user-attachments/assets/03c50feb-4d04-489d-af24-ebecc680dfd4

Here's [a workflow](https://github.com/BigStationW/ComfyUi-Image-And-Reference-To-Video/blob/main/Workflow_I2V%2BReference.json) for those interested.

## Installation

Navigate to the **ComfyUI\custom_nodes** folder, [open cmd](https://www.youtube.com/watch?v=bgSSJQolR0E&t=47s) and run:

```bash
git clone https://github.com/BigStationW/ComfyUi-MiniMax-H3-Image-And-Reference-To-Video
```
Restart ComfyUI after installation.

## Hybrid model

You need an hybrid version of MiniMax-H3 to make both I2V and reference images work:

https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models

(b20-49 works fine on my end)

## Why not simply use ([Add Guide for MiniMax H3](https://github.com/Comfy-Org/ComfyUI/blob/187eda8ef5e588c6a5765cad53e482765edae052/comfy_extras/nodes_minimax_h3.py#L162) + [MiniMax H3 Reference to Video](https://github.com/Comfy-Org/ComfyUI/blob/187eda8ef5e588c6a5765cad53e482765edae052/comfy_extras/nodes_minimax_h3.py#L239)) to get "the same thing"?

### MiniMax H3 Image + Reference to Video node (This custom node)

- The start and end frames are fed directly into the text-vision encoder (Qwen 3 VL). That way the text encoder is able to build accurate text embeddings through the understanding of the image before diffusion begins.

- The encoded VAE latents of those frames are directly injected into the diffusion transformer to physically anchor pixels across specified frames (start/end).

### Add Guide for MiniMax H3 + MiniMax H3 Reference to Video

- In ```Add Guide for MiniMax H3```, only the DiT receives the frame latents. The text encoder never sees those images. This is an issue as the text encoder will build text embeddings while being completely unaware of the scene's starting lighting, geometry, and camera angle.

- During the denoising process, the DiT is caught in a tug-of-war: the text cross-attention is pulling the generation in one direction (pure text context), while the latent injection forces it into the keyframe pixels. This mismatch manifests as color shifts, warping, unnatural transitions, flickering, etc.

Even if we set aside the technical explanation, you can see from the videos that the quality is lower when using ```Add Guide for MiniMax H3 + MiniMax H3 Reference to Video``` rather than using this custom node.

[Comparison.webm](https://github.com/user-attachments/assets/e99d216d-4dcd-42f7-ab16-29b6548e9fed)




