# DenialOps

AI-powered insurance claim denial understanding and action copilot.

## Overview

DenialOps helps patients and healthcare providers understand insurance claim denials and take appropriate action. Given a denial letter, it:

1. **Extracts** structured information (denial reason, codes, dates, amounts)
2. **Routes** to the best action path (prior auth, claim correction, appeal)
3. **Generates** personalized action plans and document templates

## Quick Start

```bash
# Install dependencies
make install

# Start the server
make run

# Run tests
make test
```

## Usage

```bash
# 1. Create a case
curl -X POST http://localhost:8000/cases \
  -H "Content-Type: application/json" \
  -d '{"mode": "fast"}'

# 2. Upload denial letter
curl -X POST http://localhost:8000/cases/{case_id}/documents \
  -F "file=@denial_letter.pdf" \
  -F "doc_type=denial_letter"

# 3. Run pipeline
curl -X POST http://localhost:8000/cases/{case_id}/run

# 4. Get action plan
curl http://localhost:8000/cases/{case_id}/artifacts/action_plan.md
```

## Operating Modes

- **Fast Mode**: Works with just a denial letter. Provides general guidance.
- **Verified Mode**: Include your SBC/EOC for grounded, cited recommendations.

## Supported Routes

| Route | When Used |
|-------|-----------|
| `prior_auth_needed` | Prior authorization missing or incomplete |
| `claim_correction_resubmit` | Coding, modifier, or billing error |
| `medical_necessity_appeal` | Medical necessity determination |

## Project Structure

```
denialops/
├── src/denialops/       # Main package
│   ├── api/             # FastAPI routes
│   ├── models/          # Pydantic models
│   ├── pipeline/        # Processing stages
│   ├── llm/             # LLM client
│   └── utils/           # Utilities
├── schemas/             # JSON schemas
├── tests/               # Test suite
├── data/samples/        # Sample denial letters
└── docs/                # Documentation
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
ANTHROPIC_API_KEY=sk-ant-xxxxx
ENVIRONMENT=dev
LOG_LEVEL=INFO
```

## Development

```bash
make lint       # Run linter
make format     # Format code
make typecheck  # Type checking
make test       # Run tests
make eval       # Run evaluation
```

## Documentation

- [Technical Architecture](docs/technical-architecture.md) - **Deep-dive for engineers** (LLM integration, pipeline design, prompt engineering)
- [Spec](docs/spec.md) - Product specification
- [Implementation Plan](docs/implementation-plan.md) - Build checklist
- [Evaluation](docs/eval.md) - Quality gates
- [Threat Model](docs/threat-model.md) - Security considerations

## Disclaimers

- This tool is for informational purposes only
- Not legal or medical advice
- Verify all information with your insurance company

## License

MIT
