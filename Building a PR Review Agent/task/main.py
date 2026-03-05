"""
Please provide the full URL to your recipes-api GitHub repository below.
"""

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from github import Github
from urllib.parse import urlparse
from typing import Any

import dotenv
import os

dotenv.load_dotenv()

repo_url = "https://github.com/BalaDakshina/recipe-api.git"

git = Github(os.getenv("GITHUB_TOKEN"))

def repo_full_name_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path


repo = git.get_repo(repo_full_name_from_url(repo_url))

print(repo.full_name)
print(repo.default_branch)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    pr = repo.get_pull(pr_number)

    commit_shas = []
    for c in pr.get_commits():
        commit_shas.append(c.sha)

    details = {
        "author": pr.user.login if pr.user else None,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "head_sha": pr.head.sha,
        "commit_shas": commit_shas,
        "number": pr.number,
        "html_url": pr.html_url,
        "base_branch": pr.base.ref,
        "head_ref": pr.head.ref,
    }
    return details


llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPEN_API_KEY"),
    api_base="https://litellm.aks-hs-prod.int.hyperskill.org/openai"
)


def multiply(num1: float, num2: float) -> float:
    var = num1 * num2
    return var

print(get_pr_details(1))


def get_pr_commit_details(commit_sha: str) -> dict[str, Any]:
    commit = repo.get_commit(commit_sha)

    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": getattr(f, "patch", None),  # can be missing/None
        })

    return {
        "sha": commit.sha,
        "html_url": commit.html_url,
        "message": commit.commit.message if commit.commit else None,
        "files": changed_files,
    }


details = get_pr_details(1)
print(get_pr_commit_details(details["commit_shas"][0]))

import base64


def get_file_contents(path: str, ref: str | None = None) -> dict[str, Any]:
    try:
        contents = repo.get_contents(path, ref=ref) if ref else repo.get_contents(path)
        if isinstance(contents, list):
            return {
                "path": path,
                "ref": ref,
                "type": "directory",
                "entries": [c.path for c in contents],
            }

        if contents.encoding == "base64" and contents.content:
            raw = base64.b64decode(contents.content).decode("utf-8", errors="replace")
        else:
            raw = contents.decoded_content.decode("utf-8", errors="replace")

        return {"path": path, "ref": ref, "type": "file", "text": raw}

    except Exception as e:
        return {"path": path, "ref": ref, "error": str(e)}


details = get_pr_details(1)
print(get_file_contents("README.md", ref=details["head_sha"]))

pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
commit_details_tool = FunctionTool.from_defaults(fn=get_pr_commit_details)
files_tool = FunctionTool.from_defaults(fn=get_file_contents)

context_agent = ReActAgent(
    llm=llm,
    tools=[pr_details_tool, commit_details_tool, files_tool],
    system_prompt="""You are the context gathering agent. When gathering context, you MUST gather \n: 
    - The details: author, title, body, diff_url, state, and head_sha; \n
    - Changed files; \n
    - Any requested for files; \n
""",
    name="ContextAgent",
)

import asyncio
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate

async def main():
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = context_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print(event.response.content)
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")

if __name__ == "__main__":
    asyncio.run(main())
    git.close()
