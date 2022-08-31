#!/usr/bin/env python3

import torch
from datasets.prepare_sequences import germanBats
from models.bat_2_js import BAT

classes = germanBats

model = BAT(
    max_len=60,
    d_model=64,
    num_classes=len(list(classes)),
    nhead=2,
    dim_feedforward=32,
    num_layers=2,
)
model.load_state_dict(torch.load('models/bat_2_convnet_mixed.pth', map_location='cpu'))
model.eval()

dummy_input = torch.zeros((1, 60, 44, 257))
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model, dummy_input, 'BAT.onnx', verbose=True, input_names=input_names, output_names=output_names)
