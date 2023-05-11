from datasets import Dataset as HFDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset


class SNIL_For_T5(Dataset):

    def __init__(self, tokenizer: T5Tokenizer) -> None:
        super().__init__()
        self.dataset = HFDataset.from_file(
            '/home/liukai/.cache/huggingface/datasets/esnli/plain_text/0.0.2/a160e6a02bbb8d828c738918dafec4e7d298782c334b5109af632fec6d779bbc/esnli-train.arrow',
            split='train[0:1000]')
        self.tokenizer = tokenizer

        self.inputs = []
        self.labels = []
        self._build()

    def _build(self):
        label_mapping = {
            0: "entailment </s>",
            1: "neutral </s>",
            2: "contradiction </s>",
        }
        print('Building SNIL_For_T5 dataset...')
        inputs = []
        labels = []
        for i, data in enumerate(self.dataset.to_list()):
            input_str = f"[label]: premise: {data['premise']}. hypothesis: {data['hypothesis']}"
            label_str = label_mapping[data['label']]

            inputs.append(input_str)
            labels.append(label_str)
            if i % 1000 == 0:
                print(f"{i}/{len(self)}")

        self.inputs = self.tokenizer.batch_encode_plus(
            inputs,
            padding='longest',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')['input_ids']
        self.labels = self.tokenizer.batch_encode_plus(
            labels,
            padding='longest',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')['input_ids']

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return {
            "input_ids": self.inputs[index].squeeze(),
            "labels": self.labels[index].squeeze(),
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        "t5-small")

    dataset = SNIL_For_T5(tokenizer)
    for data in dataset:
        out = model.forward(input_ids=data['input_ids'].unsqueeze(0),
                            labels=data['labels'].unsqueeze(0))
        print(out.loss)

        print(data)
        break