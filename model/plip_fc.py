from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from torch import nn
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config, get_peft_model


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class PLIP_fc(nn.Module):
    def __init__(self, num_class=2) -> None:
        super(PLIP_fc, self).__init__()

        model = CLIPModel.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")
        processor = CLIPProcessor.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip", TOKENIZERS_PARALLELISM=False)
        self.feature_extractor = model.vision_model
        # self.fc = nn.Sequential(nn.ReLU(), nn.Linear(768, 2))
        # self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), \
        #                         nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        # self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))
        # self.feature_extractor.requires_grad_ = False

    def forward(self, x):
        x = self.feature_extractor(x)
        # print(x)
        # x = F.relu(x)
        # print(x.pooler_output.shape)
        x = self.fc(x.pooler_output)
        return x


class PLIP_fc_frozen(nn.Module):
    def __init__(self) -> None:
        super(PLIP_fc_frozen, self).__init__()

        model = CLIPModel.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")
        self.feature_extractor = model.vision_model
        for params in self.feature_extractor.parameters():
            params.requires_grad = False
        # self.fc = nn.Sequential(nn.ReLU(), nn.Linear(768, 2))
        # self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), \
        #                         nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2))
        # self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))
        # self.feature_extractor.requires_grad_ = False

    def forward(self, x):
        x = self.feature_extractor(x)
        # print(x)
        # x = F.relu(x)
        # print(x.pooler_output.shape)
        x = self.fc(x.pooler_output)
        return x

def get_plip_fc_lora(num_class=2):
    model = PLIP_fc(num_class)
    target_modules = ["encoder.layers.{0}.self_attn.v_proj".format(i) for i in range(12)] + \
        ["encoder.layers.{0}.self_attn.q_proj".format(i) for i in range(12)]
    # target_modules = ["encoder.layers.{0}.self_attn.v_proj".format(i) for i in range(12)] + \
    #     ["encoder.layers.{0}.self_attn.q_proj".format(i) for i in range(12)] + \
    #     ["encoder.layers.{0}.self_attn.k_proj".format(i) for i in range(12)]
    # print(target_modules)

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules, # ['k_proj', 'v_proj']
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["fc"],
    )

    lora_model = get_peft_model(model, config)
    return lora_model


if __name__ == "__main__":
    # model = PLIP_fc()
    model = PLIP_fc_frozen()
    # model = get_plip_fc_lora()
    print_trainable_parameters(model)
    image = torch.rand((2,3,224,224))
    out = model(image)
    print(out.shape)
    pass