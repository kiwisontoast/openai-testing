import os
import openai
# os.system("python openai_api.py")
openai.api_key = ("<apikey>")

response = openai.Completion.create(
  engine="text-curie-001",
  prompt="What is the best fruit?",
  temperature=0.7,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response)

        