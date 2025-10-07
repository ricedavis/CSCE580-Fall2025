1) PF1 – Basic Direct Prompt

Convert the following recipe text into a JSON file that follows the R3 recipe data format (Rich Recipe Representation).
The JSON should include:

"recipe_name"

"data_provenance (include a placeholder source_url and date if missing)"

"macronutrients (can be empty {})"

"ingredients (each with name, quantity.measure, quantity.unit, allergies, alternative, quality_characteristic, image)"

"instructions (each with original_text, input_condition, task with action_name, output_quality, background_knowledge, and modality)"

"and booleans for hasDairy, hasMeat, and hasNuts."

Make sure the JSON is syntactically valid and properly nested.

Here is the recipe text:
(PASTED RECIPE TEXT HERE)


2) PF2 – Schema-Guided Prompt

I need you to transform the recipe below into a JSON file consistent with the R3 structure used in the “Egg-Drop-Chicken-Noodle-Soup” example on GitHub.

Follow this schema exactly:
(PASTED SCHEMA EXAMPLE HERE)

Then read this recipe and fill in the fields accordingly, using simple, robot-friendly English instructions (one step per task).
(PASTED RECIPE TEXT HERE)



3) PF3 – Step-Reasoning Prompt

Act as a structured-data engineer converting recipes into the R3 JSON format.
First, extract key metadata: recipe name, prep time, cook time, servings, and ingredients.
Next, identify each cooking step and represent it as an R3 instruction object that includes:

the original step text

its task(s) with action_name, output_quality, and background_knowledge

a short list of input and output conditions

Finally, combine all fields into a complete, valid R3 JSON object.
Ensure the JSON is valid and has no comments or extra explanations.

Here is the recipe text:
(PASTED RECIPE TEXT HERE)
