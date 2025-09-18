# Model Building and Risk Analysis with Health Survey Data

**Authors:**  
Sonya Allin (York University, sallin@yorku.ca)  
Lisa Zhang (University of Toronto, lczhang@cs.toronto.edu)  
Mustafa Haiderbhai (University of Toronto)  
Carolyn Quinlan (University of Toronto)  
Rutwa Engineer (University of Toronto)  
Michael Pawliuk (University of Toronto)  

---

## Overview

This project consists of a sequence of three programming labs in which students build and analyze machine learning models to predict the presence of heart disease using NHANES survey responses. Students implement and evaluate **decision tree, logistic regression, and neural network models**, while also exploring **model evaluation, subgroup fairness, and bias-variance decomposition**.  

The labs are structured to integrate **technical model building** with **ethical and risk considerations**:

1. **Lab 1:** Exploratory data analysis, decision tree classifiers, hyperparameter tuning, and reflection on model evaluation.  
2. **Lab 2:** Logistic regression, stochastic gradient descent, subgroup analysis, and consideration of error types between sensitive groups.  
3. **Lab 3:** Bias-variance decomposition using synthetic and real-world data, ensemble models, and analysis of the impact of data quantity on model error.  

---

## Assignment Information

**Topics Covered:**

- **Lab 1:** Decision Trees, Model Evaluation, Hyperparameter Tuning  
- **Lab 2:** Logistic Regression, Subgroup Analysis  
- **Lab 3:** Polynomial Regression, Bias-Variance Decomposition, Ensemble Methods  

**Audience:** Upper-level undergraduate students in a machine learning course.  

**Difficulty:** Moderate. Each lab typically takes **2–3 hours**, but can be adjusted depending on course needs.  

---

## Strengths

- Combines technical content with discussions on **ethical, legal, and human factors**.  
- Encourages critical thinking about **data collection, quality, and limitations**.  
- Illustrates potential issues in real-world applications via **subgroup analysis and fairness considerations**.  
- Uses **health data**, which is intuitively sensitive but small enough for practical training times.  

---

## Weaknesses

- Covering both **data nuances** and **technical modeling** may be challenging in fast-paced courses.  
- Reflection exercises are recommended to help students synthesize insights from the labs.  
- Some advanced tasks (e.g., SGD implementation) may need to be omitted for courses with limited time.  

---

## Dependencies

**Software:** Python with Jupyter Notebooks (compatible with Google Colab).  

**Python Packages:**
- `numpy`  
- `pandas`  
- `matplotlib`  
- `sklearn`  
- `graphviz` and `pydotplus` (Lab 1 visualizations)  

**Prior Material:**
- Lab 1 assumes knowledge of decision trees.  
- Lab 2 assumes knowledge of logistic regression.  
- Lab 3 assumes understanding of polynomial regression and bias-variance concepts.  

---

## Variants

- Not all tasks in the labs need to be graded.  
- Labs can be **simplified** by omitting tasks such as SGD implementation.  
- Different models can replace the ones provided without impacting the discussion on evaluation and subgroup analysis.  

---

## Summary of Resources

| File | Description |
|------|------------|
| `Makefile` | Uses `notedown` to generate `.ipynb` files from markdown files. |
| `lab01.md` | Lab 1 markdown file. |
| `lab01.ipynb` | Student-facing Lab 1 Jupyter Notebook. |
| `lab02.md` | Lab 2 markdown file. |
| `lab02.ipynb` | Student-facing Lab 2 Jupyter Notebook. |
| `lab03.md` | Lab 3 markdown file. |
| `lab03.ipynb` | Student-facing Lab 3 Jupyter Notebook. |
| `NHANES-heart.csv` | Subset of NHANES dataset used across all labs. |

---

## Contact

For solutions or questions, please contact the authors via the emails listed above.