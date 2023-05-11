#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from t5.snil import SNIL_For_T5

from transformers import Trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer


def train():

    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small",
        model_max_length=1024,
    )

    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        "t5-small")

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      train_dataset=SNIL_For_T5(tokenizer=tokenizer, ))
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()
