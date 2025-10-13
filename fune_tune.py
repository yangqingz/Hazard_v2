from openai import OpenAI
import os
os.environ['OPENAI_API_KEY'] ="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"

client = OpenAI()



files = client.files.list()

for file in files:
    print(f"Deleting file: {file.id}")
    client.files.delete(file.id)

client.files.create(
  file=open("train.jsonl", "rb"),
  purpose="fine-tune"
)

files = client.files.list()

for file in files:
    file_id = file.id
# List all files

# Delete each file


# print("All files deleted successfully!")
# file_id = "file-9e3K44dVdq7A5xj9sL3TA5"  # replace with your file ID
client.fine_tuning.jobs.create(
  training_file=file_id,
  model="gpt-3.5-turbo", #change to gpt-4-0613 if you have access
  method={
        "type": "supervised",
        "supervised": {"hyperparameters": {"n_epochs": 2}},
    },
)

