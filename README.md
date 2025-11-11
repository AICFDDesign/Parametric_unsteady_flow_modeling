# Parametric unsteady flow modeling method based on $\beta$-variational autoencoders and transformers


## Data
We share the flow data in [zenodo](https://zenodo.org/records/17461011).

## Code
All the codes are in the folder [cylinder](https://github.com/AICFDDesign/Parametric_unsteady_flow_modeling/tree/master/cylinder) and [flapping](https://github.com/AICFDDesign/Parametric_unsteady_flow_modeling/tree/master/flapping).

## Requirements
The code is implemented in Python 3.10. All the libraries you need are in [requirements.txt](https://github.com/AICFDDesign/Parametric_unsteady_flow_modeling/blob/master/requirements.txt)

## Run

```bash
# 1. clone repository
git clone https://github.com/AICFDDesign/Parametric_unsteady_flow_modeling.git
cd Parametric_unsteady_flow_modeling

# 2. Set up the environment
pip install -r requirements.txt

# 3. Run the main program:
cd cylinder
python main.py

