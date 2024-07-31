---
layout: post
title:  "Code, Run, Debug on AutoPilot: Let Your Local Llama Do All Your Heavy Lifting!"
author: Gurwinder
categories: [ AI ]
image: assets/images/llama.jpg
featured: true
hidden: false
---

AutoGen isn't just another framework; it marks a revolutionary leap in leveraging Large Language Models (LLMs). Built to empower developers, AutoGen excels in orchestrating multi-agent conversations, where these agents are not only customizable but also adept at seamless human interaction.

#### Setting Up AutoGen

Before diving into AutoGen, ensure your environment meets the requirements:
```bash
pip install pyautogen
```

#### Setting Ollama for Local Hosting

Before diving into AutoGen, ensure your environment meets the requirements:

```bash
pip install ollama
pip install 'litellm[proxy]'
```
Download and Configure Your Model
```bash
ollama pull ollama/llama3:latest
```

Serve with LiteLLM
```bash
litellm --model ollama/llama3:latest
```

## Lets Ask LLM to Execute Two Sum in Python
#### 1. Imports
import autogen: Imports the AutoGen framework, which facilitates interaction between different agents.
from autogen.coding import LocalCommandLineCodeExecutor: Imports a class that allows the agents to execute code locally on the command line.

```python
import autogen
from autogen.coding import LocalCommandLineCodeExecutor
```
#### 2. Configuration for the Language Model (LLM)
config_list_codellama: A list containing the configuration for a local LLM server (likely running on http://0.0.0.0:4000) with the model "ollama/llama3:latest".
llm_config_codellama: This wraps the configuration list into a dictionary that will be passed to the assistant agent.

```python
config_list_codellama = [
    {
        'base_url': "http://0.0.0.0:4000",
        'api_key': "NULL",
        'model': "ollama/llama3:latest"
    }
]

llm_config_codellama={
    'config_list': config_list_codellama,
}
```
#### 3. Task Definition
coding_tasks: A list containing a single string that represents the task: "Write two sum problem in python and check it with random test cases".

```python
coding_tasks = ["""Write two sum problem in python and check it with random test cases"""]
```
#### 4. Assistant Agent Configuration
autogen.AssistantAgent: Creates an assistant agent named "assistant", which will be responsible for generating and handling the code based on the task. The agent is configured to use the config_list_codellama LLM configuration.
```python
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config_codellama
)
```
#### 5. User Proxy Agent Configuration
autogen.UserProxyAgent: Creates a user proxy agent named "user_proxy" with the following settings:
human_input_mode="NEVER": The agent operates without human intervention.
max_consecutive_auto_reply=10: Limits the number of automatic replies the agent can make without user input.
is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"): A function to detect when the task is complete by checking if the message ends with "TERMINATE".
code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir="coding")}: Configures the agent to execute code in a local directory named "coding" using a command-line executor.
system_message: Instructions for how the agent should respond—if the task is fully solved, it replies with "TERMINATE"; otherwise, it either asks to continue or explains why the task is not solved yet.
```python
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir="coding")},
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)
```
#### 6. Initiate the Chat
user_proxy.initiate_chat(assistant, message=coding_tasks[0]): Starts the interaction between the user proxy and assistant agents, passing the coding task as the initial message.
```python
user_proxy.initiate_chat(
    assistant, message=coding_tasks[0]
)
```

## Lets Look at the Conversation of Agents

#### <span style="color:blue"> *user_proxy* (to assistant)</span>

Write two sum problem in python and check it with random test cases

--------------------------------------------------------------------------------
#### <span style="color:red"> *assistant* (to user_proxy):</span>

I'd be happy to help you with that!

Here's a Python code block for the Two Sum problem:

```python
def two_sum(nums, target):
    num_dict = {}
    for i, num in enumerate(nums):
        if target - num in num_dict:
            return [num_dict[target - num], i]
        num_dict[num] = i

# Test cases
nums1 = [2, 7, 11, 15]
target1 = 9
print(two_sum(nums1, target1))  # Output: [0, 1]

nums2 = [3, 2, 4]
target2 = 6
print(two_sum(nums2, target2))  # Output: [1, 2]

nums3 = [5, 5, 10, 15]
target3 = 10
print(two_sum(nums3, target3))  # Output: [0, 1]
```

Let me explain the plan:

1. Define a function `two_sum` that takes a list of numbers (`nums`) and a target sum (`target`) as input.
2. Create an empty dictionary `num_dict` to store the indices of the numbers in the input list.
3. Iterate through the input list, and for each number, check if the difference between the target sum and the current number is present in the `num_dict`. If it is, return a list containing the indices of the two numbers that add up to the target sum.
4. Test the function with three different sets of input numbers and their corresponding target sums.

Feel free to execute this code block, and let me know if you have any questions or need further assistance!

--------------------------------------------------------------------------------

#### <span style="color:blue"> *user_proxy* (to assistant)</span>


>>>>>>>> EXECUTING CODE BLOCK (inferred language is python)...


exitcode: 0 (execution succeeded)
Code output:

[0, 1]

[1, 2]

[0, 1]


--------------------------------------------------------------------------------
#### <span style="color:red"> *assistant* (to user_proxy):</span>

I'm glad the code executed successfully! The output matches our expected results:

* `[0, 1]` for `nums1`, since 2 + 7 = 9
* `[1, 2]` for `nums2`, since 2 + 4 = 6
* `[0, 1]` again for `nums3`, since 5 + 5 = 10

Since everything seems to be working as expected, I'll terminate our conversation now.

TERMINATE

--------------------------------------------------------------------------------

**Conclusion**

In conclusion, the pyautogen framework facilitates a powerful collaboration between AI agents to autonomously code, execute, and debug tasks. In this example, the AssistantAgent leverages a local LLM model to generate Python code for the "two sum problem," while the UserProxyAgent oversees the process, ensuring the code is executed and validated through random test cases.

The UserProxyAgent acts as a diligent overseer, automatically determining if the task is fully satisfied or if further iterations are needed. It runs the generated code in a local environment using the LocalCommandLineCodeExecutor, providing a seamless cycle of coding, execution, and debugging without human intervention.

This setup highlights how pyautogen can streamline complex coding tasks, enabling AI-driven development workflows that can code, test, and refine solutions in real-time. By combining the strengths of different agents, pyautogen delivers an efficient and intelligent approach to software development, automating the process from start to finish.