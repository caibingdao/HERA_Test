# Evaluation visualization index

Visualizations are organized first by action-selection strategy and seed
count, then by model.

```text
visualization/
├── greedy_1_seed_4777/
│   ├── ariadne_original/
│   ├── ariadne_mamba/
│   ├── ariadne_mamba_3_1/
│   │   ├── pure_policy/
│   │   ├── recovery_v4/
│   │   └── comparison_pure_vs_recovery_v4/
│   └── cross_model_comparison/
├── greedy_5_seeds_100_104/
│   ├── ariadne_original/
│   ├── ariadne_mamba/
│   ├── ariadne_mamba_3_1/
│   │   ├── pure_policy/
│   │   ├── recovery_v4/
│   │   └── comparison_pure_vs_recovery_v4/
│   └── cross_model_comparison/
├── stochastic_1_seed_4777/
│   ├── ariadne_original/
│   ├── ariadne_mamba/
│   ├── ariadne_mamba_3_1/
│   │   ├── pure_policy/
│   │   └── recovery_v4/
│   └── cross_model_comparison_pure_policy/
└── stochastic_5_seeds_100_104/
    ├── ariadne_original/
    ├── ariadne_mamba/
    ├── ariadne_mamba_3_1/
    │   ├── pure_policy/
    │   └── recovery_v4/
    └── cross_model_comparison_pure_policy/
        ├── failure_cases_refined/
        └── failure_cases_legacy/
```

- `pure_policy`: policy actions only; Loop Recovery V4 was disabled.
- `recovery_v4`: the Mamba 3.1 policy with Loop Recovery V4 enabled and able
  to override the policy when revisit/low-progress conditions trigger.
- `cross_model_comparison_pure_policy`: comparisons generated before recovery
  was connected to evaluation. These compare only the three pure-policy runs.
- `failure_cases_refined`: the final side-by-side failure trajectory pages.
- `failure_cases_legacy`: the earlier trajectory rendering, retained for
  traceability.
