"""Framework-neutral evidence core.

Holds the evidence schema (plus its Pydantic mirror), structuring (markdown -> JSON), the
find-then-verify grounding cascade (presence / MiniCheck entailment / guarded LLM-finder), and the
audited claim-entailment benchmark. Carries no dependency on any agent framework.
"""
