"""
Sales script framework — system prompt template + industry templates (T09).

Plan reference: Sections 6.1, 8.3.

Architecture
------------
build_system_prompt() fills the STATIC slots (lead data, company, opener,
objection scripts) at call-start and returns a string that still contains
the DYNAMIC placeholder tokens literally:

    {conversation_summary}   — filled by CallMemory.build_context()
    {last_utterance}         — filled by CallMemory.build_context()
    {objections_list}        — filled by CallMemory.build_context()
    {decision}               — filled by CallMemory.build_context()

This two-pass design keeps build_system_prompt() a pure function (no
memory/state dependency) while letting the context window stay current.

FCC 24-17 compliance
---------------------
AI disclosure is mandatory in every call opener.  All IndustryTemplate
opener_text strings include a disclosure phrase.  Never remove it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Master system prompt template ─────────────────────────────────────────────
# Static slots: filled once by build_system_prompt().
# Dynamic slots: literal {tokens} left for CallMemory.build_context() to fill.

SYSTEM_PROMPT = """\
You are {agent_name}, a sales representative at {company_name}.
You are currently on a live phone call with {lead_name}, who runs {business_name}
in the {industry} industry in {city}, {state}.

PERSONA: {persona_description}
Speak naturally — short sentences, conversational, confident but not pushy.
Maximum 35 words per response. Never use bullet points or lists. Speak like a human.

CALL OBJECTIVE: {objective}

OPENING SCRIPT:
"{opener_text}"

VALUE PROPOSITION (1-2 sentences max):
{value_prop}

PROBE QUESTIONS (ask these naturally during conversation):
{probe_questions}

TALKING POINTS (use when relevant):
{talking_points}

OBJECTION HANDLING:
If they say budget/cost → "{objection_budget}"
If they say timing/not now → "{objection_timing}"
If they say not interested → "{objection_not_interested}"
If they say they have a solution → "{objection_competitor}"
If they want you to call back → "{objection_callback}"

CLOSE ATTEMPT (after value delivered + 1 objection handled):
"{close_text}"

GRACEFUL EXIT:
If they decline firmly → Thank them, wish them well, hang up cleanly.
If they want a callback → Get their preferred time, confirm, thank them.
If they want info sent → Get email, confirm, thank them.

CONVERSATION STATE (current):
{conversation_summary}
Last prospect utterance: "{last_utterance}"
Objections raised so far: {objections_list}
Decision reached: {decision}

RULES:
- Never lie about who you are or what your company does
- Never pressure after two clear declinations
- Always disclose you are an AI if directly asked (FCC 24-17 — mandatory)
- End every response with a question or clear next step
- If confused or lost → "Could you repeat that?"
"""

# ── AIA objection scripts (Acknowledge / Incentivize / Ask) ───────────────────
# Plan reference: Section 8.3

OBJECTION_SCRIPTS: dict[str, str] = {
    "budget": (
        "That makes sense — most businesses are watching costs closely right now. "
        "What's interesting is similar companies said the same thing before they started — "
        "they actually freed up budget by [specific area] in about six weeks. "
        "Is budget the only thing holding you back, or is there something else?"
    ),
    "timing": (
        "Totally understand — timing is everything. "
        "The reason I'm reaching out now is that [specific trigger]. "
        "I'm not asking for a decision today, just a 15-minute look. "
        "What would make this the right time for you?"
    ),
    "not_interested": (
        "Completely fair — you weren't expecting this call. "
        "The reason I reached out specifically was [specific reason]. "
        "If [specific outcome] wasn't relevant to you I wouldn't have called. "
        "What would have to be true for this to make sense for your business?"
    ),
    "competitor": (
        "Great — that means you've already thought about this. "
        "Can I ask — what made you choose your current solution? "
        "And what's the one thing you wish it did differently? "
        "That gap is exactly what we built for. Worth a quick look?"
    ),
    "callback": (
        "Absolutely, I can work around your schedule. "
        "To make sure I actually reach you — would [day] at [time] work, "
        "or is there a better window later this week?"
    ),
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LeadContext:
    """
    Minimal lead data needed to build a system prompt.
    Populated from the Lead ORM model (T43) or manually in tests.
    """
    lead_name: str
    business_name: str
    industry: str          # key into INDUSTRY_TEMPLATES or a free-form string
    city: str
    state: str
    talking_points: str = ""   # from lead-scoring enrichment (T43)


@dataclass
class ScriptContext:
    """
    Call-level configuration — who is calling and why.
    Populated from Settings + campaign config at call-start.
    """
    agent_name: str
    company_name: str
    persona_description: str = "professional, confident, and genuinely helpful"
    objective: str = "book a 15-minute discovery call"


@dataclass(frozen=True)
class IndustryTemplate:
    """Per-industry script fragments injected into the system prompt."""
    industry: str
    opener_text: str      # MUST include AI disclosure (FCC 24-17)
    value_prop: str
    probe_questions: str
    close_text: str


# ── Industry templates (5 minimum per plan) ───────────────────────────────────
# FCC 24-17: every opener_text must identify the call as AI-generated.

INDUSTRY_TEMPLATES: dict[str, IndustryTemplate] = {
    "real_estate": IndustryTemplate(
        industry="real_estate",
        opener_text=(
            "Hi {lead_name}, this is {agent_name} from {company_name} — "
            "quick heads-up, I'm an AI assistant calling on their behalf. "
            "We work with real estate agents in {city} to automate lead follow-up "
            "and book more listing appointments without extra ad spend. "
            "Do you have 60 seconds?"
        ),
        value_prop=(
            "We helped a real estate team in {state} automate their follow-up "
            "and increase listing appointments by 40% in 90 days — "
            "without hiring another person."
        ),
        probe_questions=(
            "Are you currently doing any automated outreach to your pipeline, "
            "or is most of it manual calls and texts? "
            "And how many leads a month are slipping through because of slow follow-up?"
        ),
        close_text=(
            "I'd love to show you what we built for agents in your market. "
            "Would Tuesday at 2pm or Thursday at 10am work "
            "for a quick 15-minute call?"
        ),
    ),

    "healthcare": IndustryTemplate(
        industry="healthcare",
        opener_text=(
            "Hi {lead_name}, {agent_name} from {company_name} — "
            "I'm an AI calling on their behalf, so just a quick disclosure there. "
            "We help medical practices in {city} reduce no-shows and fill "
            "last-minute appointment gaps automatically. Do you have a minute?"
        ),
        value_prop=(
            "We helped a clinic in {state} cut their no-show rate by 35% "
            "and recover about $4,000 a month in missed appointments — "
            "fully automated, no extra staff."
        ),
        probe_questions=(
            "What's your current no-show rate looking like, and "
            "how are you reminding patients today — automated texts, calls, or manual?"
        ),
        close_text=(
            "I'd like to walk you through how this works for a practice your size. "
            "Are you free Wednesday at 11am or Friday at 2pm "
            "for a 15-minute screen share?"
        ),
    ),

    "home_services": IndustryTemplate(
        industry="home_services",
        opener_text=(
            "Hi {lead_name}, this is {agent_name} with {company_name} — "
            "and just to be upfront, I'm an AI calling on their behalf. "
            "We help {industry} businesses in {city} fill their schedule "
            "with qualified local jobs without paying more per lead. "
            "Got 60 seconds?"
        ),
        value_prop=(
            "We helped a contractor in {state} add 12 new recurring customers "
            "in their first 60 days — using their existing customer list, "
            "no extra ad budget."
        ),
        probe_questions=(
            "How are you currently getting new jobs — referrals, ads, something else? "
            "And in a slow week, what's the biggest gap you're trying to fill?"
        ),
        close_text=(
            "I'd love to show you the exact playbook we used for them. "
            "Would Monday morning or Wednesday afternoon work "
            "for a quick 15-minute call?"
        ),
    ),

    "finance": IndustryTemplate(
        industry="finance",
        opener_text=(
            "Hi {lead_name}, {agent_name} here from {company_name}. "
            "Quick disclosure — I'm an AI assistant calling on their behalf. "
            "We work with financial advisors in {city} to automate prospect "
            "outreach and increase AUM without cold-calling manually. "
            "Do you have 60 seconds?"
        ),
        value_prop=(
            "We helped an advisor in {state} add $2.3M in new AUM in six months "
            "by re-engaging their dormant prospect list — "
            "fully compliant, no manual dialing."
        ),
        probe_questions=(
            "How many prospects in your CRM haven't heard from you in over 90 days? "
            "And what's your typical process for re-engaging them?"
        ),
        close_text=(
            "I'd love to show you the compliance-friendly workflow we built for them. "
            "Are you free Thursday at 1pm or Friday at 3pm "
            "for a 15-minute walkthrough?"
        ),
    ),

    "tech_saas": IndustryTemplate(
        industry="tech_saas",
        opener_text=(
            "Hi {lead_name}, {agent_name} calling from {company_name} — "
            "I'm an AI assistant, just so you know. "
            "We help SaaS companies in {city} reduce churn and expand revenue "
            "from their existing accounts automatically. "
            "Do you have a quick minute?"
        ),
        value_prop=(
            "We helped a B2B SaaS company in {state} reduce churn by 18% "
            "and grow NRR to 115% in one quarter — "
            "without adding headcount to their CS team."
        ),
        probe_questions=(
            "What does your current churn rate look like, and "
            "how are you identifying at-risk accounts before they cancel?"
        ),
        close_text=(
            "I'd like to walk you through how this maps to your stack. "
            "Would Tuesday at 3pm or Thursday at 11am work "
            "for a 15-minute technical overview?"
        ),
    ),
}

_GENERIC_TEMPLATE = IndustryTemplate(
    industry="generic",
    opener_text=(
        "Hi {lead_name}, this is {agent_name} from {company_name} — "
        "just to be upfront, I'm an AI calling on their behalf. "
        "We help businesses in {city} grow more efficiently. "
        "Do you have 60 seconds?"
    ),
    value_prop=(
        "We've helped companies similar to yours in {state} "
        "achieve measurable results in their first 90 days."
    ),
    probe_questions=(
        "What's the biggest growth challenge you're working on right now?"
    ),
    close_text=(
        "I'd love to show you how this could work for {business_name}. "
        "Would sometime this week work for a quick 15-minute call?"
    ),
)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_industry_template(industry: str) -> IndustryTemplate:
    """
    Return the template for `industry`, normalising the key.
    Falls back to the generic template when no match is found.
    """
    normalised = industry.lower().replace(" ", "_").replace("-", "_")
    # Try exact match first, then prefix match
    if normalised in INDUSTRY_TEMPLATES:
        return INDUSTRY_TEMPLATES[normalised]
    for key, tpl in INDUSTRY_TEMPLATES.items():
        if key in normalised or normalised in key:
            return tpl
    return _GENERIC_TEMPLATE


class _SafeFormat(dict):
    """dict subclass that returns '{key}' for missing keys in str.format_map()."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def build_system_prompt(
    lead: LeadContext,
    script: ScriptContext,
    industry_tpl: IndustryTemplate | None = None,
) -> str:
    """
    Fill the static slots in SYSTEM_PROMPT and return the result.

    Dynamic slots (conversation_summary, last_utterance, objections_list,
    decision) are intentionally left as literal {tokens} for
    CallMemory.build_context() to inject at each turn.

    Parameters
    ----------
    lead:         Prospect data (name, company, location, industry).
    script:       Agent/company configuration.
    industry_tpl: Pre-fetched industry template; resolved via
                  get_industry_template(lead.industry) when None.
    """
    tpl = industry_tpl or get_industry_template(lead.industry)

    # Render the per-industry snippets with lead/script placeholders
    sub = _SafeFormat(
        lead_name=lead.lead_name,
        business_name=lead.business_name,
        industry=lead.industry,
        city=lead.city,
        state=lead.state,
        agent_name=script.agent_name,
        company_name=script.company_name,
    )
    opener = tpl.opener_text.format_map(sub)
    value_prop = tpl.value_prop.format_map(sub)
    probe_questions = tpl.probe_questions.format_map(sub)
    close_text = tpl.close_text.format_map(sub)
    talking_points = lead.talking_points or "Use industry knowledge and the value proposition above."

    # Build the full context dict for SYSTEM_PROMPT.
    # Dynamic slots (conversation_summary, last_utterance, objections_list,
    # decision) are NOT present → _SafeFormat.__missing__ leaves them as
    # literal {tokens} for CallMemory.build_context() to inject each turn.
    ctx = _SafeFormat(
        agent_name=script.agent_name,
        company_name=script.company_name,
        lead_name=lead.lead_name,
        business_name=lead.business_name,
        industry=lead.industry,
        city=lead.city,
        state=lead.state,
        persona_description=script.persona_description,
        objective=script.objective,
        opener_text=opener,
        value_prop=value_prop,
        probe_questions=probe_questions,
        talking_points=talking_points,
        close_text=close_text,
        objection_budget=OBJECTION_SCRIPTS["budget"],
        objection_timing=OBJECTION_SCRIPTS["timing"],
        objection_not_interested=OBJECTION_SCRIPTS["not_interested"],
        objection_competitor=OBJECTION_SCRIPTS["competitor"],
        objection_callback=OBJECTION_SCRIPTS["callback"],
    )
    return SYSTEM_PROMPT.format_map(ctx)
