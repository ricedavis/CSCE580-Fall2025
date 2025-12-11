# CSCE 580 – Final Project

This repository contains all code, data, and written responses for the CSCE 580 final project, which focuses on building an AI-based attendance audit system from handwritten class attendance sheets.

---

## Repository Structure

.
├── code/
│   ├── master_csv.py
│   └── analysis.ipynb
├── data/
│   └── attendance_master.csv
├── Q1a_final.pdf
├── Q2_Responses.pdf
└── README.md


## Contents Overview

### 1. `code/` Directory
This folder contains all source code used for processing, consolidating, and analyzing the attendance dataset.

#### `master_csv.py`
A Python script responsible for:
- Loading all extracted JSON files (one per class session)
- Consolidating them into a single structured dataset
- Generating the final CSV output (`attendance_master.csv`)

This script forms the core data-aggregation pipeline for the project.

#### `analysis.ipynb`
A Jupyter Notebook used to:
- Load the consolidated CSV
- Perform statistical analysis using Pandas
- Answer all required questions in **Q2, Part C**, including:
  - Number of classes and dates  
  - Median attendance  
  - Lowest/highest attendance  
  - Correlation between attendance and evaluation dates  

This notebook is the central analysis component of the project.

---

### 2. `data/` Directory

#### `attendance_master.csv`
A single, unified CSV dataset containing all attendance records across all 27 class sessions.  
Each row corresponds to a single student entry, including:
- Class number  
- Class date  
- Serial number  
- Full name  
- Username  

This file is the direct output of `master_csv.py` and serves as the input to `analysis.ipynb`.

---

### 3. PDF Documents

#### `Q1a_final.pdf`
A written response to **Question 1 (Part A)** of the final exam.  
This file contains the explanation and critique of a graduate-level presentation based on the assigned paper.

#### `Q2_Responses.pdf`
A complete set of written answers for **Question 2**, including:
- Data preparation methods  
- Model creation and justification  
- Analytical results  
- Interpretation of attendance patterns  
- Proposed improvements with additional time  

This PDF contains all narrative responses required for the final submission.

---

## Purpose of the Repository

This repository demonstrates the end-to-end workflow for using AI-based image extraction and Python analytics to audit classroom attendance. It includes:

- Automated text extraction (performed externally via GPT vision tools)  
- Data consolidation and cleaning  
- Exploratory attendance analysis  
- Formal written documentation of findings  

The project highlights the integration of LLM-powered perception, Python-based data science, and structured academic reporting.

---

## How to Use the Repository

1. **Run `master_csv.py`** (if re-creating the master dataset from JSON files).  
2. **Open `analysis.ipynb`** to run all attendance analyses.  
3. **Refer to `Q2_Responses.pdf`** for narrative explanations.  
4. **Use `attendance_master.csv`** as the clean, final dataset for any additional analysis.

---

If you would like enhancements such as:
- A “How to Reproduce” section  
- Installation or environment setup  
- A license section  
- Shields.io badges  

Just let me know!

