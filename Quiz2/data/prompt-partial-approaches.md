PP-1: Extract Ingredients

Prompt:
“You are a cooking assistant. I will provide you a recipe text. Extract only the ingredients from the recipe and convert them into R3 JSON format. For each ingredient, include:

name

quantity (split into measure and unit)

allergies (use empty values if unknown)

alternative (optional substitutes)

quality_characteristic (any descriptors)

image (leave blank)
Output only valid JSON with an ingredients key containing a list of ingredient objects. Do not include instructions yet.”

PP-2: Extract Instructions

Prompt:
“You are a cooking assistant. I will provide you a recipe text. Extract only the step-by-step instructions from the recipe and convert them into R3 JSON format. For each instruction, include:

original_text (the text of the step)

input_condition (what ingredients or items are needed at this step, if applicable)

task (list of actions in the step with action_name, output_quality, background_knowledge)

output_condition (what is achieved after the step)

modality (leave image and video empty)
Output only valid JSON with an instructions key containing a list of instruction objects. Do not include ingredients or metadata yet.”
