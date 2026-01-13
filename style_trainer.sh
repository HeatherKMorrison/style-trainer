#!/usr/bin/env bash

corpora=( corpora/*.json )
valid=false

while [[ $valid == false ]]; do
	yesno=false
	echo "Which corpus would you like to fine-tune with?"
	i=0
	for corpus in "${corpora[@]}"; do
		name=${corpus##*/}
		echo "$i: $name"
		((i++))
	done
	int=false
	while [[ $int == false ]]; do
	read -p "Please enter a number:" number
		if [[ -v corpora[$number] ]]; then
			int=true
		else
			echo "Invalid input."
		fi
	done
	echo "You selected ${corpora[$number]}."
	while [[ $yesno == false ]]; do
		read -p "Is this correct? (y)es or (n)o: " confirm
		shopt -s nocasematch
		if [[ $confirm == y* ]]; then
			yesno=true
			valid=true
		elif [[ $confirm == n* ]]; then
			yesno=true
		else
			echo "Invalid input"
		fi
	done
	
done

valid=false

while [[ $valid == false ]]; do
	read -p "Is this a new adapter? (y)es or (n)o: " new
	fresh=true
	shopt -s nocasematch
	if [[ $new == y* ]]; then
		valid=true
				
	elif [[ $new == n* ]]; then
		fresh=false
		valid=true
	else
		echo "Invalid Selction"
				
	fi
done

valid=false

while [[ $valid == false ]]; do
	read -p "How many total steps would you like to train? Please enter a whole number: " steps
	if [[ $steps =~ ^-?[0-9]+$ ]]; then
		valid=true
		
	else
		echo "Invalid Selction"
	
	fi
done

python style_trainer.py $fresh "${corpora[$number]}" $steps
