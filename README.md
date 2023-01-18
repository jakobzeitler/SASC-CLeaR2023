# SASC-CLeaR2023

```
pronounced 'sassy' or 'SAS-C'
```

Run with:
```
python main_SASC.py
```
Based on _Non-parametric identifiability and sensitivity analysis of synthetic
control models_, published in CLeaR (Causal Learning and Reasoning) 2023 (https://www.cclear.cc/2023)




## Packages

pandas, scitkit-learn, matplotlib, tabulate

Install from requirements.txt:
```
conda create --name SASC --file requirements.txt
```
## Data

### California Prop99

See code Shi et al.'s paper _On the Assumptions of Synthetic Control Methods_: https://github.com/claudiashi57/fine-grained-SC

### German Reunification 
See code from Shi et al.'s code from "Theory for identification and Inference with Synthetic Controls: A Proximal Causal Inference Framework"

In line 73 of "Application_GermanReunification.R" run:

'
export <- data.frame(data.all$Y,data.all$W,data.all$X)
write.csv(export, "GermanReunificationGDP.csv", row.names = FALSE)
'