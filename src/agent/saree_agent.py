import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.config import Config
from src.agent.tools import search_similar_sarees

# Multi-version import support for LangChain Agents
try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    try:
        from langchain.agents.tool_calling.base import create_tool_calling_agent
        from langchain.agents import AgentExecutor
    except ImportError:
        from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

class SareeAgent:
    def __init__(self):
        Config.validate()
        
        self.llm = None
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.2
            )
        except Exception:
            pass
        
        self.tools = [search_similar_sarees]
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert Indian saree stylist and visual search assistant. "
                "Help users find visually similar sarees based on color, pattern, border, pallu, and weave. "
                "Always call the `search_similar_sarees` tool when the user provides an image or asks for recommendations. "
                "When returning search results, present each item clearly with its Name, Price, Match Score, and Website Link."
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        if self.llm:
            try:
                agent = create_tool_calling_agent(self.llm, self.tools, prompt)
                self.executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
            except Exception:
                self.executor = None
        else:
            self.executor = None

    def run(self, user_input: str) -> str:
        """Executes query through LLM agent with fail-safe fallback to direct visual search."""
        # Try running through LLM Agent
        if self.executor:
            try:
                response = self.executor.invoke({"input": user_input})
                output = response.get("output", "")
                if output:
                    return output
            except Exception as e:
                print(f"LLM API error ({e}). Executing direct vector search fallback...")

        # FAIL-SAFE: Direct Qdrant Visual Search Execution
        image_match = re.search(r"\[Query Image Source:\s*(.*?)\]", user_input)
        image_source = image_match.group(1).strip() if image_match else None

        if not image_source:
            return "Please upload a saree image or enter an image URL to find visually similar items."

        results = search_similar_sarees.invoke({"image_input": image_source, "top_k": 5})

        if not results or (isinstance(results, list) and len(results) > 0 and "error" in results[0]):
            err_msg = results[0].get("error", "Unknown error") if results else "No matches found"
            return f"Could not perform visual search: {err_msg}"

        output_lines = ["### Top Matching Sarees Found:\n"]
        for idx, item in enumerate(results, start=1):
            payload = item.get("payload", {})
            score = item.get("score", 0)
            name = payload.get("name", "Saree")
            price = payload.get("discounted_price") or payload.get("retail_price") or "N/A"
            link = payload.get("website_link", "#")
            
            output_lines.append(f"**{idx}. {name}**")
            output_lines.append(f"- **Match Score:** {int(score * 100)}%")
            output_lines.append(f"- **Price:** ₹{price}")
            output_lines.append(f"- [View Details & Buy]({link})\n")

        return "\n".join(output_lines)