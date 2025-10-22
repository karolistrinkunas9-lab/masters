# masters
ORIGINAL DATASETS:
(DS1/DS2/DS3/DS4/SmellyWordsDictionary) https://github.com/m-zakeri/ARTA/ ARTA tool

TRAINING DATASETS:
Text classification AI: x_train_32_strat_801010.csv, x_validation_32_strat_801010.csv, x_verification_32_strat_801010.csv
Correction quality AI: synthetic.jsonl

The idea, is to assist in the process of requirements engineering, by creating a system to modify requirements.
The core steps include the evaluation process, the detection of issues (before and after modification) and the modification of the requirement.

Base software system functionality:
  1. Ability to detect requirement quality issues (trained AI model)
  2. Ability to assist in evaluating software requirement quality, and quality of requirement modification
  3. Ability to suggest changes and modify the target requirement based on feedback

TEXT CLASSIFICATION (all-MiniLM-L6-v2):
  Input (X) -> 
    "If any part of the call fails, an audible and visual indication shall be provided in the appropriate cab."
  Output (Y) -> 
    Predicted labels: ['Nonverifiable']
    Probabilities: [0.394, 0.021, 0.99, 0.012, 0.017]

CHANGE QUALITY (Qwen/Qwen3-0.6B):
  INPUT (X) -> 
    {
      "role": "user", 
      "content": "
        Original: The system shall send an email notification to users after a password reset.\n
        Modification: Adjust behavior to use in-app notifications instead of email.\n
        Modified: The system shall display an in-app notification to users after a password reset.
      "
    }
  Output (Y) ->
  [
    {'generated_text': 
      [
        {
          'role': 'user', 
          'content': 
            'Original: The system shall send an email notification to users after a password reset.\n
            Modification: Adjust behavior to use in-app notifications instead of email.\n
            Modified: The system shall display an in-app notification to users after a password reset.
          '
        }, 
        {
          'role': 'assistant', 
          'content': '<think>\n\n</think>\n\n
          {
            "preservation_correctness": 0.87, 
            "change_correctness": 0.54, 
            "comments": [
              "Original behavior preserved",
              "Access limited to in-app notifications"
            ]
          }
        '}
      ]
    }
  ]


Example:
Input 1 (Original requirement): The system should support three languages - lithuanian, english and danish
Input 2 (Feedback): The system does not support danish
Output: The system should support two languages - lihuanian and english


INPUT OUTPUT FOR AI MODEL:

INPUT (X) -> Requirement
OUTPUT (Y) -> Issues with requirement

Example 1:
Input (Requirement): Users should be able to access most of the system when authorized.
Output: Ambiguity

Example 2:
Input (Requirement):  A special indication should be shown to the user
Output: Subjective language

Example 3:
Input (Requirement):  The driver shall select an override control according to the permission received
Output: -

