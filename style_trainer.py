from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

#Set up pretrained model
model_name = "EleutherAI/pythia-2.8b" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize(token):
    text = [i + " " + o for i, o in zip(token["input"], token["output"])]

    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )

#load and tokenize corpus
dataset = load_dataset("json", data_files="fantasy_mini-corpus.json", split="train")#chainge data_files to the name of the corpus
tokens = dataset.map(tokenize, batched=True)
tokens = tokens.map(
    lambda x: {"labels": x["input_ids"]}, batched=True
)
#Split for validation
split = tokens.train_test_split(test_size=0.2)
training = split["train"]
testing = split["test"]

#set up LoRA
config = LoraConfig( 
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

#Confgiure the model with LoRA
lora_model = get_peft_model(model, config)

#Set up training
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-4,
    max_steps=200,
    #num_train_epochs=3,
    logging_dir="./logs",
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    fp16=True,
    push_to_hub=False
)
trainer = Trainer(
    model=lora_model,
    args=args,
    train_dataset=training,
    eval_dataset=testing
)

#fine tune
trainer.train()

#save adapter
lora_model.save_pretrained(r"C:\Users\callu\Documents\Job_Fantasy\adapters\fantasy_adapater")#choose best name for the adapter

