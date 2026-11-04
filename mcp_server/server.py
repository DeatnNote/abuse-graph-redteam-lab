"""
MCP server: lets an analyst (or an LLM agent) query the abuse graph in plain language.

  python -m mcp_server.server         # stdio; builds world 0 and the v1 detector on start
"""
try:                                                  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:                                   # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server

from agl.detector import Detector, split_groups
from agl.sim import generate
from mcp_server.graph_tools import GraphStore

server = _Server("abuse-graph")
_store = None


def store():
    global _store
    if _store is None:
        df = generate(0)
        _store = GraphStore(df, Detector().fit(df, split_groups(df)))
    return _store


@server.tool()
def account_risk(account_id: str) -> dict:
    """Risk score, flag and cluster id for one advertiser account."""
    return store().account_risk(account_id)


@server.tool()
def cluster_summary(account_id: str) -> dict:
    """The operator cluster an account belongs to: size, flags, shared (masked) attributes, members."""
    return store().cluster_summary(account_id)


@server.tool()
def top_clusters(n: int = 10) -> list:
    """Clusters with the most flagged accounts."""
    return store().top_clusters(n)


if __name__ == "__main__":
    server.run()
