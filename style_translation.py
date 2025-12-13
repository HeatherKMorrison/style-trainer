from transformers import GPTNeoXForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch


#Load adapted model
adapter_path = r"C:\Users\callu\Documents\Job_Fantasy\adapters\fantasy_adapater"

config = PeftConfig.from_pretrained(adapter_path)
torch.set_default_dtype(torch.float32)
basic_model = GPTNeoXForCausalLM.from_pretrained(config.base_model_name_or_path, low_cpu_mem_usage=False, dtype=torch.float32)
adapter_model = PeftModel.from_pretrained(basic_model, adapter_path)

if torch.cuda.is_available():
    adapter_model = adapter_model.to(torch.device("cuda"))
    adapter_model = adapter_model.to(torch.float16)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)


#Build pipeline
new_text = pipeline(
    "text-generation",
    model=adapter_model,
    tokenizer=tokenizer,
    device=0
)
#Prompt results
text_input = "Rewrite 3+ years python as a fantasy quest."
output = new_text(text_input, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id )

#Print results
print(output[0]["generated_text"])