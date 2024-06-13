from torch import nn

from transformers import ViTModel, ViTPreTrainedModel


class ViTClassifier(ViTPreTrainedModel):
    def __init__(self, config):
        super(ViTClassifier, self).__init__(config)

        self.vit = ViTModel(config)

        self.batch_norm = nn.BatchNorm2d(1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

        """
        for name, param in self.vit.named_parameters():
            if "classifier" not in name:  # Freeze layers that are not the classifier
                param.requires_grad = False
                """

    def forward(self, pixel_values):
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_output = vit_outputs.pooler_output

        vit_output = self.batch_norm(vit_output)
        logits = self.classifier(vit_output)

        return logits


if __name__ == "__main__":
    model = ViTClassifier.from_pretrained(
        "google/vit-large-patch32-384",
    )
    print(model)
