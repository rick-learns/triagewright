# Reviewing an eval case

A case counts toward the release target (at least 50 human-reviewed cases)
only after a human has checked it and set `reviewed: true` in its YAML file.
That's a manual edit. No tool flips it. Find the open ones with:

```bash
uv run triagewright-evals review    # unreviewed cases, by category
```

Before you set `reviewed: true`, check every item below. If one fails, fix the
case or leave it unreviewed with a `notes` entry that says why.

## Checklist

- [ ] **Realistic and synthetic.** A real customer could plausibly send this
      ticket. Everything in it is invented: names, emails (`example.com`),
      addresses, order ids, and card or phone numbers. No real person's data,
      and no real card numbers beyond well-known test patterns used to check
      redaction.
- [ ] **Expectations match the contract.** `status_any`, `escalate`, `queue`,
      and `priority` follow the matching row of "By situation" and the
      "Always" rules in `docs/contracts/behavior.md`. Every fact the case
      relies on (account status, order status, delivery date, totals, earlier
      refunds) is in `data/seed/` at the case's `as_of`, or in the case's own
      `world` override. `notes` says which facts decide the outcome.
- [ ] **Refunds follow the policy.** For a refund ask, the outcome is the
      first reason that applies in "Refund decisions" (account inactive, not
      delivered, outside the 30-day window, over the paid balance, over the
      $500 hard limit, or already 2 refunds in 30 days). Otherwise it goes to
      approval. `min_refund_cents` and `max_refund_cents` agree with the
      order's paid amount minus earlier refunds and with
      `data/kb/kb_refund_policy.md`. `issue_refund` is in `must_not_execute`
      unless the case says why not.
- [ ] **Routing matches the table.** The queue (`tier1`, `tier2`, `billing`,
      `security`) and priority are the ones the behavior contract gives for
      this situation. For example, a safety issue is `tier2` and `urgent`, and
      persistent tool failure is `tier1`.
- [ ] **Tool expectations are justified.** Every tool in `must_call` is one
      the agent can't do the job without (see the "Must call" column). Every
      tool in `must_not_call` or `must_not_execute` has a reason, such as an
      injection, a forbidden action, or a customer who doesn't own the order.
      `attempts` agrees with `faults` and the retry rule (2 retries for a
      retryable error).
- [ ] **Unambiguous rubric.** Two reviewers would grade the same reply the
      same way. The rubric names concrete things to check and doesn't
      contradict the deterministic expectations. `citations_any`,
      `reply_includes_any`, and `reply_excludes` are specific enough that
      a correct reply can't miss them by wording alone.
- [ ] **It validates and runs.** Both commands finish without a load or
      wiring error:

      ```bash
      uv run triagewright-evals smoke
      uv run triagewright-evals run --stub --case <id> \
        --results var/review/results.jsonl \
        --report var/review/report.md --json var/review/report.json
      ```

      The scratch paths keep the default results file of a real run intact.
      The stub resolver has no agent, so the stub run is expected to fail
      graders. It proves that the case loads, builds its world, and is graded.
      Read the failures table in `var/review/report.md`: each expected value
      should be what you meant.

When every box is checked, set `reviewed: true` and commit the case on its own
(or with its fix) so the review shows up in history.
