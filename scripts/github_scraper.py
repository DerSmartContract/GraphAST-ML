import requests
import os
import json

def get_github_python_repos(query="machine learning", max_repos=5):
    """
    Holt öffentliche Python-Repositories von GitHub anhand eines Suchbegriffs.
    """
    url = f"https://api.github.com/search/repositories?q={query}+language:python&sort=stars&per_page={max_repos}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        repos = response.json().get("items", [])
        return [(repo["name"], repo["clone_url"]) for repo in repos]
    else:
        print("❌ Fehler beim Abruf der GitHub-Repositories!")
        return []

def download_python_files(repo_name, repo_url, save_dir="data/raw"): 
    """
    Klont ein GitHub-Repository und speichert die Python-Dateien lokal.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 Lade Repository {repo_name} herunter...")
    os.system(f"git clone --depth 1 {repo_url} temp_repo")
    
    for root, _, files in os.walk("temp_repo"):
        for file in files:
            if file.endswith(".py"):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(save_dir, f"{repo_name}_{file}")
                os.rename(src_path, dest_path)
                print(f"✔️ {file} gespeichert als {dest_path}")
    
    os.system("rm -rf temp_repo")

def main():
    print("🔍 Suche nach Python-Repositories...")
    repos = get_github_python_repos(query="machine learning", max_repos=3)
    
    for repo_name, repo_url in repos:
        download_python_files(repo_name, repo_url)
    
    print("✅ GitHub-Daten erfolgreich heruntergeladen!")

if __name__ == "__main__":
    main()
