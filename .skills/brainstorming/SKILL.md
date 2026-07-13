---
name: brainstorming
description: Explore and shape ambiguous product, content, UI, architecture, or feature ideas before implementation. Use when the user asks to brainstorm, design, scope, compare approaches, create a spec, or when requirements are materially unclear and implementation would require risky assumptions.
---

# Brainstorming

Use this skill to turn unclear ideas into an implementable design. Keep the process proportional: a small change may need only a short design note; a new product or workflow needs a fuller spec.

## Workflow

1. Inspect project context first: files, docs, existing patterns, and recent relevant work.
2. Clarify the goal, constraints, audience, success criteria, and non-goals. Ask one question at a time when user input is required.
3. If the request spans multiple independent systems, decompose it before designing details.
4. Present 2-3 viable approaches with tradeoffs and a recommendation.
5. Present the selected design at the right level of detail: architecture, data flow, UI/UX, error handling, testing, and rollout when relevant.
6. Ask for approval before implementation when the design materially affects behavior, UX, architecture, or scope.
7. Write a spec only when the user asks for one or when the work is large enough that implementation would be error-prone without it.

## Spec Guidance

Default spec path:

```text
docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md
```

A useful spec includes:

- Problem and goals.
- Non-goals.
- Current system context.
- Proposed design.
- User-visible behavior.
- Data/model changes, if any.
- Error states and edge cases.
- Test plan.
- Rollout or migration notes, if needed.

Before handing off a spec, scan for placeholders, contradictions, ambiguous requirements, and scope creep. Fix issues inline.

Commit a spec only when the user asks for a commit or the repo workflow explicitly requires it for the task. Do not push without approval.

## Visual Companion

If upcoming brainstorming would be clearer with visual mockups, diagrams, or side-by-side options, offer the browser companion in its own message:

```text
Some of what we're working on might be easier to explain if I can show it to you in a web browser. I can put together mockups, diagrams, comparisons, and other visuals as we go. This feature is still new and can be token-intensive. Want to try it? (Requires opening a local URL)
```

Use the companion only for visual questions: UI layouts, diagrams, spatial flows, or visual comparisons. Use normal conversation for requirements, scope, tradeoffs, and technical choices.

If the user accepts, read `.skills/brainstorming/visual-companion.md` before starting the companion.

## Design Principles

- Prefer existing project patterns over new abstractions.
- Keep scope narrow and explicit.
- Separate units by clear responsibilities and interfaces.
- Include targeted cleanup only when it directly supports the requested work.
- Make success criteria testable.
- Avoid implementation work until the user has approved the design when approval is needed.
