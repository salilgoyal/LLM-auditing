from huggingface_hub import login, hf_hub_download, list_repo_files

# Define the directory where files will be saved
save_dir = "/Users/salilgoyal/Stanford/LLM-auditing/chatbot-arena"

# Define the remote repository ID
repo_id = "lmarena-ai/chatbot-arena-leaderboard"

files = list_repo_files(repo_id=repo_id, repo_type="space")
files = [file for file in files if file.endswith('pkl') or file.endswith('csv')]

# Download files
for file in files:
    file_path = hf_hub_download(repo_id=repo_id, filename=file, repo_type="space", local_dir=save_dir)