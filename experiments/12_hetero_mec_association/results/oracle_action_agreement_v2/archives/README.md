# Oracle v2 raw artifacts (compressed for GitHub)

The uncompressed per-step and per-episode files are far above GitHub's 100 MB
per-file limit (137 MB and 141 MB per group). They are stored here as
compressed archives; the uncompressed originals remain under
`results/oracle_action_agreement_v2/<group>/`.

Per group:

- `steps.part00.jsonl.gz` — one JSON line per decision step, including all
  Oracle v2 audit fields (teacher action, selected node/model/action,
  P(a*), agreement, candidate counts, latency fields, tie-break flag).
- `episodes.tar.gz` — one JSON file per task with the full episode record and
  `trajectory` (the same step records plus episode-level metrics).

Decompress:

```bash
cd experiments/12_hetero_mec_association/results/oracle_action_agreement_v2
gunzip -c archives/two_homo/steps.part00.jsonl.gz > /tmp/two_homo_steps.jsonl
mkdir -p /tmp/two_homo_episodes
tar -xzf archives/two_homo/episodes.tar.gz -C /tmp/two_homo_episodes
```

`SHA256SUMS.txt` lists the archive hashes.
