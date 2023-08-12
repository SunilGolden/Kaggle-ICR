# ICR - Identifying Age-Related Conditions

## Challenge Link
**Link:** https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview

## Data
**Link:** https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data

<br />

## Usage Guide

### Install Dependencies

1. **Install Python:** Make sure Python is installed on your system. If not, you can download and install Python from the official Python website: https://www.python.org/downloads/

2. **Create a virtual environment:** 

	```bash
	python -m venv env
	```

3. **Activate the virtual environment**

	> For Windows
	```bash
	env\Scripts\activate
	```

	> For macOS/Linux
	```bash
	source env/bin/activate
	```

4. **Install the dependencies**
	
	```bash
	pip install -r requirements.txt
	```

<br />

### Model Evaluation

```bash
python model_evaluation.py 
```

<br />

### Training and Inference

```bash
python train_and_inference.py 
```