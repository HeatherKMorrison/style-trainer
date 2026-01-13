$corpora = @(Get-ChildItem -Path ".\corpora" -Filter "*.json")
$valid = $false
while (-not $valid){
	Write-Host "Which corpus would you like to fine-tune with?"

	for ($i = 0; $i -lt $corpora.Count; $i++){
		Write-Host "${i}: $($corpora[$i].Name)"
	}
	$corpus = [int](Read-Host "Please enter a number:")
	if ($corpus -ge 0 -and $corpus -lt $corpora.Count){
		Write-Host "You selected $($corpora[$corpus].Name)"
		$yesno = $false
		while (-not $yesno){
			$correct = Read-Host "Is this correct? (y)es or (n)o"
			if ($correct.StartsWith("Y", [System.StringComparison]::OrdinalIgnoreCase)){
				$valid = $true
				$yesno = $true
			} elseif ($correct.StartsWith("N", [System.StringComparison]::OrdinalIgnoreCase)){
				$yesno = $true
			} else {
				Write-Host "Invalid Selction"
				Start-Sleep -Seconds 1
			}
		}
	}
	else {
		Write-Host "Invalid Selction"
		Start-Sleep -Seconds 1
	}
}

$valid = $false

while(-not $valid){
	$new = Read-Host "Is this a new adapter? (y)es or (n)o"
	$fresh = $true
	if ($correct.StartsWith("Y", [System.StringComparison]::OrdinalIgnoreCase)){
				$valid = $true
				
			} elseif ($correct.StartsWith("N", [System.StringComparison]::OrdinalIgnoreCase)){
				$fresh = $false
				$valid = $true
			} else {
				Write-Host "Invalid Selction"
				Start-Sleep -Seconds 1
			}
}

$valid = $false

while(-not $valid){
	$steps = (Read-Host "How many total steps would you like to train? Please enter a whole number")
	try{
		$total = [int]$steps
		$valid = $true
		
	} catch {
		Write-Host "Invalid Selction"
		Start-Sleep -Seconds 1
	}
}

python style_trainer.py $fresh $(Join-Path "corpora" $corpora[$corpus].Name) $total