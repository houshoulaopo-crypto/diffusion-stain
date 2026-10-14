import argparse
import os
import yaml
import torch
from torchsummary import summary
from models.ncsnv2 import NCSNv2Deeper, NCSNv2Deeper_Light_Weight

# 将 dict 转为支持属性访问的 config 对象
class ConfigObject:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = ConfigObject(v)
            setattr(self, k, v)

def load_config(path):
    with open(path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    return ConfigObject(raw_cfg)

def main():
    print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='bedroom.yml')
    args = parser.parse_args()

    config_path = os.path.join('configs', args.config)
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    config.device = device
    model = NCSNv2Deeper_Light_Weight(config).to(device)
    model.eval()


    x = torch.randn(4, config.data.channels,64, 64).to(device)
    y = torch.randint(0, config.model.num_classes, (4,)).to(device)

    with torch.no_grad():
        output = model(x, y)

    print("Output shape:", output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数数量: {total_params}")
    print(f"可训练参数数量: {trainable_params}")

if __name__ == '__main__':
    main()
