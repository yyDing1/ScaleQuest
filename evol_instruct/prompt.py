base_instruction = """
You are an Instruction Rewriter.
Your objective is to rewrite a given instruction into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
But the rewritten instruction must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Instruction#:. Also, please do not omit the input in #Given Instruction#.
You SHOULD complicate the given instruction using the following method:
If #Given Instruction# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.
You should try your best not to make the #Rewritten Instruction# become verbose, #Rewritten Instruction# can only add 10 to 20 words into #Given Instruction#.

Output Format:

#Given Instruction#: [Given instruction]
#Rewritten Instruction#: [Rewritten instruction]

#Given Instruction#: {}
"""

base_instruction2 = """
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
"""




# def createConstraintsPrompt(instruction):
# 	prompt = base_instruction.format("Please add one more constraints/requirements into #Given Prompt#'")
# 	prompt += "\n#Given Prompt#:\n{}\n".format(instruction)
# 	prompt += "\n#Rewritten Prompt#:\n"
# 	return prompt

# def createDeepenPrompt(instruction):
# 	prompt = base_instruction.format("If #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
# 	prompt += "\n#Given Prompt#:\n{}\n".format(instruction)
# 	prompt += "\n#Rewritten Prompt#:\n"
# 	return prompt

# def createConcretizingPrompt(instruction):
# 	prompt = base_instruction.format("Please replace general concepts with more specific concepts.")
# 	prompt += "\n#Given Prompt#:\n{}\n".format(instruction)
# 	prompt += "\n#Rewritten Prompt#:\n"
# 	return prompt


def createHardEvolPrompt(problem):
	# prompt = base_instruction.format(
	# 	"If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.",
	# 	instruction.strip()
	# )
	prompt = base_instruction2.format(
		problem=problem
	)
	return prompt

# def createEasyEvolPrompt(instruction):
# 	prompt = base_instruction.format("If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")
# 	prompt += "\n#Given Prompt#:\n{}\n".format(instruction.strip())
# 	prompt += "\n#Rewritten Prompt#:\n"
# 	return prompt

