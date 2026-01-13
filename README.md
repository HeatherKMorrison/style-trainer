# Style Trainer  
A user friendly tool for fine-tuning the output of Pythia 2.8b to unique styles using LoRA.  

## Purpose  

The purpose of this tool is to provide a simplified means of fine-tuning a pretrained LLM's output style. This requires little more than a basic understanding of command line and python.

## Requirements  

This project was created with Python 3.12.3. It can be downloaded from here:  

https://www.python.org/downloads/  

The following additional packages are required:  

transformers  
peft  
datasets  

They can be installed using requirements.txt:  

pip install -r requirements.txt  

## Quickstart  

Using a command line terminal navigate to the folder containing this tool and corpus files.  

Powershell:  
cd path\to\project 

Bash:  
cd path/to/project  

Run the script and follow the prompts to begin or resume training.  

Powershell:  
powershell -executionpolicy bypass -File .\style_trainer.ps1 

Bash:  
bash style_trainer.sh
