from openai import OpenAI
import os
os.environ['OPENAI_API_KEY'] =""

client = OpenAI()


# client.files.create(
#   file=open("train2.jsonl", "rb"),
#   purpose="fine-tune"
# )

files = client.files.list()
for file in files:
    print(file)
# List all files

# # Delete each file
# for file in files:
#     print(f"Deleting file: {file.id}")
#     client.files.delete(file.id)

print("All files deleted successfully!")
file_id = "file-9e3K44dVdq7A5xj9sL3TA5"  # replace with your file ID
client.fine_tuning.jobs.create(
  training_file=file_id,
  model="gpt-3.5-turbo", #change to gpt-4-0613 if you have access
  method={
        "type": "supervised",
        "supervised": {"hyperparameters": {"n_epochs": 2}},
    },
)

