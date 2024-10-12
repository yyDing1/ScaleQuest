solvability_optimization_prompt = """
You are an Math Problem Rewriter that rewrites the given #Problem# into a more complex version.
Please follow the steps below to rewrite the given "#Problem#" into a more complex version.

Step 1: Please read the "#Problem#" carefully and list all the possible methods to make this problem more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Note that the problem itself might be erroneous, and you need to first correct the errors within it.

Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Problem# more complex. The plan should include several methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Rewritten Problem#. #Rewritten Problem# can only add 10 to 20 words into the "#Problem#".

Step 4: Please carefully review the #Rewritten Problem# and identify any unreasonable parts. Ensure that the #Rewritten Problem# is only a more complex version of the #Problem#. Just provide the #Finally Rewritten Problem# without any explanation and step-by-step reasoning guidance.

Please reply strictly in the following format:
Step 1 #Methods List#:
Step 2 #Plan#:
Step 3 #Rewritten Problem#:
Step 4 #Finally Rewritten Problem#:

#Problem#:
{problem}
""".strip()

difficulty_optimization_prompt = """
Please act as a professional math teacher.
Your goal is to create high quality math word problems to help students learn math.
You will be given a math question. Please optimize the Given Question and following instructions.
To achieve the goal, please follow the steps:
# Please check that the given question is a math question and write detailed solution to the Given Question.
# Based on the problem-solving process, double check the question is solvable.
# If you feel that the given question is not a meaningful math question, rewrite one that makes sense to you. Otherwise, modify the Given question according to your checking comment to ensure it is solvable and of high quality.
# If the question can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.

You have five principles to do this:
# Ensure the optimized question only asks for one thing, be reasonable and solvable, be based on the Given Question (if possible), and can be answered with only a number (float or integer). For example, DO NOT ask, 'what is the amount of A, B and C?'.
# Ensure the optimized question is in line with common sense of life. For example, the amount someone has or pays must be a positive number, and the number of people must be an integer.
# Ensure your student can answer the optimized question without the given question. If you want to use some numbers, conditions or background in the given question, please restate them to ensure no information is omitted in your optimized question.
# Please DO NOT include solution in your question.

Given Question: {problem}
Your output should be in the following format:
CREATED QUESTION: <your created question>
VERIFICATION AND MODIFICATION: <solve the question step-by-step and modify it to follow all principles>
FINAL QUESTION: <your final created question>
""".strip()


def createSolvabilityPrompt(problem):
	prompt = solvability_optimization_prompt.format(
		problem=problem
	)
	return prompt


def createDifficultyPrompt(problem):
	prompt = difficulty_optimization_prompt.format(
		problem=problem
	)
	return prompt
