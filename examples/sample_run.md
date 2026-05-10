# Example run

```bash
$ python -m agent.cli "What are the tradeoffs of SQLite vs Postgres for a small SaaS?"

╭─ Research Agent ──────────────────────────────────────╮
│ Topic: What are the tradeoffs of SQLite vs Postgres   │
│        for a small SaaS?                              │
│ Depth: standard  Max iterations: 12                   │
╰───────────────────────────────────────────────────────╯
Using groq / llama-3.3-70b-versatile

⠋ Researching...

╭─ Summary ─────────────────────────────────────────────╮
│ For small SaaS applications, SQLite and PostgreSQL    │
│ each have clear roles. SQLite is a single-file        │
│ embedded database with zero operational overhead and  │
│ excellent read performance, making it ideal for       │
│ low-write-volume apps and prototypes [1][2]. It       │
│ struggles, however, with concurrent writes — the      │
│ entire database is locked per write transaction [1].  │
│                                                       │
│ PostgreSQL adds operational complexity (a separate    │
│ server process, backups, tuning) but offers true      │
│ row-level concurrency, mature replication, and        │
│ advanced features like JSON indexing and full-text    │
│ search [3]. For a SaaS expecting >100 concurrent      │
│ writers or rich querying needs, Postgres is the       │
│ safer default [3][4]...                               │
╰───────────────────────────────────────────────────────╯

Key findings
  • SQLite has zero operational overhead but serializes writes
  • Postgres scales horizontally with read replicas
  • SQLite + WAL mode handles surprising load for read-heavy apps
  • Postgres beats SQLite for full-text search and JSON queries
  • Migration from SQLite -> Postgres is straightforward early on

Sources
  [1] https://www.sqlite.org/whentouse.html
  [2] https://blog.example.com/sqlite-in-production
  [3] https://www.postgresql.org/about/featurematrix/
  [4] https://news.example.com/postgres-vs-sqlite-2024

Open questions
  ? Specific write-throughput numbers vary widely by hardware
  ? WAL mode and litestream change SQLite's calculus significantly

╭─ Run stats ───────────╮
│ Iterations:    8       │
│ Notes:         11      │
│ Sources:       4       │
│ Cost (USD):    $0.0000 │  ← Groq free tier
╰────────────────────────╯
```

## What this teaches you

Watch the trace (`--trace`) and you'll see the agent:

1. Search for "SQLite vs Postgres tradeoffs SaaS"
2. Fetch the top 2 results
3. Take 3-4 notes per page
4. Search again for a more specific angle (concurrent writes, scaling)
5. Fetch 1-2 more pages
6. Decide it has enough and call `finish`

Then notice the failure modes you'll inevitably hit:

- **Repeated searches**: caught by the `searched_queries` guard
- **Pages that 403 or time out**: the model gets the error and tries another URL
- **Hallucinated facts**: the `take_note` requirement of `source_url` makes this harder
- **Run-on iterations**: the `max_iterations` cap and fallback `_finish` save you

These are the real lessons. The framework on the infographic is fine — but
running into the loop's failure modes is what teaches you to build for production.
