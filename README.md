# TuneKit

TuneKit/
├── tunekit/
│   ├── __init__.py           # Main package exports
│   ├── state.py              # TuneKitState TypedDict
│   ├── schemas.py            # AgentDecision (Pydantic)
│   ├── tools/
│   │   ├── __init__.py       # Tool exports
│   │   ├── ingest.py         # ingest_data
│   │   ├── validate.py       # validate_quality
│   │   ├── analyze.py        # analyze_dataset
│   │   ├── model_rec.py      # get_model_recommendation
│   │   └── config.py         # get_training_config
│   └── agents/
│       ├── __init__.py       # Agent exports
│       └── planning.py       # planning_agent
├── test_data/
│   ├── intent_data.csv       # Classification test
│   ├── qa_data.csv           # Instruction tuning test
│   └── news_data.csv         # Classification test
├── test_pipeline.py          # Test script
└── test.ipynb                # Original notebook (keep for reference)