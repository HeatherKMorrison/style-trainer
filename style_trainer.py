from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


model_name = "EleutherAI/pythia-2.8b" #Set up pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


dataset = load_dataset("json", data_files="fantasy_mini-corpus.json") #load corpus

#set up LoRA
#Confgiure the model with LoRA
#fine tune model

