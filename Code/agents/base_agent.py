"""
agents/base_agent.py
---------------------
Abstract base class for all Phase 3 investigation agents.

Every agent must implement:
    investigate(incident, parsed_logs) -> RCASignal

Agents are stateless — same inputs always produce the same output.
Agents do NOT modify incidents or logs.
Agents do NOT call each other — the agent_runner orchestrates all calls.

To add a new agent:
    1. Create agents/my_agent.py
    2. Inherit from BaseAgent
    3. Implement investigate()
    4. Register in core/agent_runner.py
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from core.incident import Incident
from agents.rca_signal import RCASignal


class BaseAgent(ABC):
    """
    Abstract base class for all Protocol Zero investigation agents.

    Subclasses must set:
        name        : str  — short identifier used in logs and RCASignal.agent_name
        description : str  — one-line description of what this agent does

    Subclasses must implement:
        investigate(incident, parsed_logs) -> RCASignal
    """

    name:        str = "base"
    description: str = "Abstract base agent"

    @abstractmethod
    def investigate(
        self,
        incident: Incident,
        parsed_logs: List[Dict],
    ) -> RCASignal:
        """
        Investigate a single incident using the full parsed log corpus.

        Args:
            incident    : The Incident object to investigate.
            parsed_logs : Full list of parsed log dicts for the current dataset.
                          These are the same dicts produced by parser.py or
                          bgl_parser.py — see context doc for exact schema.

        Returns:
            A single RCASignal with findings, hypothesis, and recommendations.
        """
        ...

    def __repr__(self) -> str:
        return f"<Agent: {self.name}>"