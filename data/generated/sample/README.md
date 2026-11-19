# Generated sample (seed 7)

Synthetic data from `triagewright.datagen`. Regenerate it with:

```bash
uv run python -m triagewright.datagen --seed 7 --customers 20 --tickets 30 \
  --out data/generated/sample
```

The same seed always writes byte-identical files, and
`tests/datagen/test_generator.py` fails if this sample drifts from the generator.

| File | What |
| --- | --- |
| `customers/accounts/orders/refunds/escalations.json` | A world for `triagewright.data.load_world`; merge it over the seed with `merge_world` |
| `meta.json` | The world clock (`as_of`, the seed's clock) and the generator arguments |
| `tickets.jsonl` | One `contracts.tickets.Ticket` per line |
| `expected.jsonl` | What each ticket was built to exercise. Refund requests record the policy outcome (`eligible` or one `RefundReason`), the reasons `evaluate_refund` returns, and the status and queue from `docs/contracts/behavior.md` |

Everything is synthetic: names come from a fixed invented list, emails use the
`.invalid` TLD, phone numbers are in the fictional 555-0100..555-0199 range, and
streets and towns are invented. Ids carry a `gen` marker (`cus_gen00001`,
`ord_gen000001`, `rf_gen000001`, `tkt_gen00001`), so they never collide with
`data/seed`. Refund outcomes assume the default `PolicyConfig` ($500 cap,
30-day window, 2 refunds per 30 days). `expected.jsonl` is a starting point for
drafting eval cases, not a set of eval cases: review each one before you use it.
