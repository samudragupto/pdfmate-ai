"""CI/CD Offline Agent Integration Smoke Runner."""

from __future__ import annotations

import asyncio

from tests.integration.smoke_test import run_smoke_verification

if __name__ == "__main__":
    asyncio.run(run_smoke_verification())
    print("CI/CD offline agent smoke test succeeded.")
