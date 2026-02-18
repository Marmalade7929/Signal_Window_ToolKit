# Publication Policy (Private -> Public Mirror)

## Purpose
Define minimum controls for promoting work from the private source repository to a public portfolio mirror.

## Allowed Public Content
- General-purpose algorithms and abstractions.
- Synthetic datasets and generated examples.
- Generic documentation and usage examples.
- Tests that do not depend on proprietary artifacts.

## Forbidden Public Content
- Real work datasets or captures.
- Vendor/customer/private identifiers.
- Internal incident notes, risk logs, or private roadmap material.
- Proprietary naming that should remain private.
- Personally identifiable information (PII).
- API secrets, tokens, or credentials.

## Required Evidence Before Publish
1. Passing CI on the export candidate.
2. Completed `governance/publish_checklist.md`.

## Retention
- Keep completed checklist state and CI evidence for traceability.
- Keep export reports as release evidence artifacts.
