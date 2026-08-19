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
