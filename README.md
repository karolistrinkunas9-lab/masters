<img width="415" height="61" alt="image" src="https://github.com/user-attachments/assets/c0aa6959-e503-4db0-9026-a32a3f466453" /># masters

The idea, is to assist in the process of requirements engineering, by creating a system to modify requirements.
The core steps include the evaluation process, the detection of issues (before and after modification) and the modification of the requirement.

Base software system functionality:
  1. Ability to detect requirement quality issues (trained AI model)
  2. Ability to assist in evaluating software requirement quality, and quality of requirement modification
  3. Ability to suggest changes and modify the target requirement based on feedback


     
INPUT OUTPUT FOR SYSTEM:
Input (X) -> Original requirement, Wanted modification
Output (Y) -> Modified requirement

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

Example 2:
Input (Requirement):  The driver shall select an override control according to the permission received
Output: -

