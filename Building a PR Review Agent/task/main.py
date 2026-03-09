"""
Please provide the full URL to your recipes-api GitHub repository below.
"""
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from github import Github
from urllib.parse import urlparse
from typing import Any
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult

import asyncio
import dotenv
import os

dotenv.load_dotenv()

repo_url = "https://github.com/BalaDakshina/recipe-api.git"
git = Github(os.getenv("GITHUB_TOKEN_PERSONAL"))


def repo_full_name_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path


repo = git.get_repo(repo_full_name_from_url(repo_url))


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


def get_pr_changed_files(pr_number: int) -> list[dict[str, Any]]:
    pr = repo.get_pull(pr_number)
    out = []
    for f in pr.get_files():
        out.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": getattr(f, "patch", None),
        })
    return out


def post_review_to_pr(pr_number: int, comment: str) -> dict[str, Any]:
    pr = repo.get_pull(pr_number)
    review = pr.create_review(body=comment, event="COMMENT")
    return {
        "posted": True,
        "review_id": review.id,
        "html_url": review.html_url,
    }

async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    async with ctx.store.edit_state() as s:
        s["state"]["draft_comment"] = draft_comment
    return "Saved draft_comment to state."


async def add_context_to_state(ctx: Context, key: str, value: Any) -> str:
    async with ctx.store.edit_state() as s:
        s["state"]["gathered_contexts"][key] = value
    return f"Saved context under key={key}"


async def add_final_review_to_state(ctx: Context, value: str) -> str:
    async with ctx.store.edit_state() as s:
        s["state"]["final_review"] = value
    return f"Saved final_review to state"


pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
commit_details_tool = FunctionTool.from_defaults(fn=get_pr_commit_details)
files_tool = FunctionTool.from_defaults(fn=get_pr_changed_files)
add_context_tool = FunctionTool.from_defaults(fn=add_context_to_state)
add_comment_tool = FunctionTool.from_defaults(fn=add_comment_to_state)
add_final_review_tool = FunctionTool.from_defaults(fn=add_final_review_to_state)
post_review_tool = FunctionTool.from_defaults(fn=post_review_to_pr)

context_agent = FunctionAgent(
    llm=llm,
    tools=[pr_details_tool, commit_details_tool, files_tool, add_context_tool],
    system_prompt="""
You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
""",
    description="Agent to collect context of the PR and share the details to Commentor PR",
    name="ContextAgent",
    can_handoff_to=["CommentorAgent"]
)

commentor_system_prompt = """
You are the commentor agent that gathers context for the given PR ALWAYS USING ContextAgent as a human reviewer would. \n 

Rules:
 - You MUST ALWAYS Request for the PR details, changed files and repo files from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - Once you have the draft Call add_comment_to_state(draft_comment=<full markdown review>) EXACTLY ONCE. \n
 - Immediately call handoff(to_agent="ReviewAndPostingAgent", reason="Draft saved; ready to finalize and post.") \n
 - Do NOT provide a final user-facing answer yourself.
 """
comment_agent = FunctionAgent(
    llm=llm,
    tools=[add_comment_tool],
    system_prompt=commentor_system_prompt,
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    name="CommentorAgent",
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

POST_REVIEW_PROMPT = """
You are the Review and Posting agent. You must use the ContextAgent to gather changes in PR and use CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 read draft_comment from state, finalize, save final_review, then call post_review_to_pr
 When you are satisfied, post the review to GitHub.  
 """

post_review_agent = FunctionAgent(
    llm=llm,
    tools=[post_review_tool, add_final_review_tool],
    system_prompt=POST_REVIEW_PROMPT,
    description="Uses the draft pr review and modifies if necessary and post it in the pr",
    name="ReviewAndPostingAgent",
    can_handoff_to=["CommentorAgent"]
)

workflow_agent = AgentWorkflow(
    agents=[post_review_agent, context_agent, comment_agent],
    root_agent=post_review_agent.name,
    initial_state={
        "gathered_contexts": {},
        "draft_comment": "",
        "final_review": ""
    },
)

ctx = Context(workflow_agent)


async def main():
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(user_msg=prompt.format(), ctx=ctx)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
