from transformers import GPTNeoXForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from transformers import logging
import requests
import torch

#Key and email
access = {"User-Agent": "user@example.com", "Authorization-Key": "<key>"}

#job title or type
confirm = False
while (confirm == False):
    job = input("How do you aim to earn your coin, adventurer? ")
    s_job = job.strip()
    print(s_job.capitalize() + ". Shall it be so?")
    aye_nay = False
    while (aye_nay == False):
        reply = input("(A)ye or (N)ay: ").lower()
        if reply.startswith("a") or reply.startswith("n"):
            aye_nay = True
        if reply.startswith("a"):
            confirm = True
            r_job = s_job.replace(" ", "+")
url = f"https://data.usajobs.gov/api/Search?Keyword={r_job}"

response = requests.get(url, headers=access)
postings = response.json()
items = postings['SearchResult']['SearchResultItems']
ad = items[0]['MatchedObjectDescriptor']
title = ad['PositionTitle']
content = ad['UserArea']['Details']

summary = content.get('JobSummary', 'The full description must be read.')
duties = content.get('Responsibilities', 'The duties will be made clear.')
qualifications = content.get('QualificationSummary', 'Any applicant will do.')
benefits = content.get('Benefits', 'The reward is in the job itself.')

text_input = title + " \n" + summary + " \n" + duties + " \n" + qualifications + " \n" + benefits


#Load adapted model
adapter_path = r"adapters\corpora\fantasy_adapter" #Windows

logging.set_verbosity_error()
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

output = new_text(text_input,min_new_tokens=50, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id )

#Print results
fantasy_text = output[0]["generated_text"]
print(title)
print(fantasy_text[len(text_input):].strip())
