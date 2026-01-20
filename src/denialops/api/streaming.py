"""Streaming API endpoints for real-time response delivery."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from denialops.api.dependencies import CaseIdDep, SettingsDep, StorageDep
from denialops.config import get_settings
from denialops.llm import create_streaming_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming"])


class StreamPromptRequest(BaseModel):
    """Request for streaming LLM completion."""

    prompt: str = Field(..., description="The prompt to send to the LLM")
    system: str | None = Field(None, description="Optional system prompt")
    max_tokens: int = Field(4096, description="Maximum tokens in response")
    temperature: float = Field(0.0, description="Temperature for sampling")


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    """Format data as a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/stream/completion")
async def stream_completion(
    request: StreamPromptRequest,
    settings: SettingsDep,
) -> StreamingResponse:
    """
    Stream an LLM completion response.

    Uses Server-Sent Events (SSE) to deliver chunks in real-time.
    Events:
    - chunk: Content chunk from the LLM
    - done: Final event with usage statistics
    - error: Error occurred during streaming
    """
    if not settings.has_llm_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM API key not configured",
        )

    async def generate_events() -> AsyncIterator[str]:
        try:
            client = create_streaming_client(
                provider=settings.llm_provider,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
            )

            response = await client.stream_async(
                prompt=request.prompt,
                system=request.system,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            full_content = ""
            async for chunk in response.chunks:
                if chunk.content:
                    full_content += chunk.content
                    yield format_sse_event("chunk", {"content": chunk.content})

                if chunk.is_final and chunk.usage:
                    yield format_sse_event(
                        "done",
                        {
                            "content": full_content,
                            "usage": {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                                "latency_ms": round(chunk.usage.latency_ms, 2),
                                "estimated_cost": round(chunk.usage.estimated_cost, 6),
                            },
                            "model": response.model,
                            "provider": response.provider,
                        },
                    )

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield format_sse_event("error", {"message": str(e)})

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/cases/{case_id}/stream/summary")
async def stream_summary(
    case_id: CaseIdDep,
    storage: StorageDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """
    Stream a personalized summary for a case.

    Prerequisites: Case must have completed the pipeline with case_facts.json available.

    Events:
    - start: Summary generation started
    - chunk: Content chunk
    - done: Summary complete with metadata
    - error: Error occurred
    """
    if not settings.has_llm_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM API key not configured",
        )

    # Load case facts
    facts_data = storage.get_artifact(case_id, "case_facts.json")
    if not facts_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Case facts not found. Run the pipeline first.",
        )

    # Load route decision if available
    route_data = storage.get_artifact(case_id, "route.json")

    async def generate_events() -> AsyncIterator[str]:
        yield format_sse_event("start", {"case_id": case_id, "status": "generating"})

        try:
            client = create_streaming_client(
                provider=settings.llm_provider,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
            )

            # Build prompt for personalized summary
            denial_reason = (
                facts_data.get("denial_reason", {}).get("stated_reason", "Unknown")
                if isinstance(facts_data.get("denial_reason"), dict)
                else "Unknown"
            )

            route_type = route_data.get("route", "appeal") if route_data else "appeal"

            system_prompt = """You are a compassionate healthcare advocate helping patients
understand their insurance claim denials. Provide clear, actionable summaries that empower
patients to take action. Be encouraging but honest about the situation."""

            user_prompt = f"""Based on the following case information, provide a personalized
summary that helps the patient understand their situation and next steps.

Denial Reason: {denial_reason}
Recommended Route: {route_type}
Service Date: {facts_data.get("dates", {}).get("service_date", "Unknown") if isinstance(facts_data.get("dates"), dict) else "Unknown"}
Provider: {facts_data.get("providers", {}).get("main_provider", "Unknown") if isinstance(facts_data.get("providers"), dict) else "Unknown"}

Provide:
1. A clear explanation of why the claim was denied (2-3 sentences)
2. What the recommended action means in practical terms
3. The most important next step to take
4. Words of encouragement

Keep the response conversational and supportive, around 200-300 words."""

            response = await client.stream_async(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=1024,
                temperature=0.7,
            )

            full_content = ""
            async for chunk in response.chunks:
                if chunk.content:
                    full_content += chunk.content
                    yield format_sse_event("chunk", {"content": chunk.content})

                if chunk.is_final and chunk.usage:
                    # Store the generated summary
                    storage.store_artifact(
                        case_id,
                        "streamed_summary.txt",
                        full_content,
                    )

                    yield format_sse_event(
                        "done",
                        {
                            "content": full_content,
                            "stored_as": "streamed_summary.txt",
                            "usage": {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            },
                        },
                    )

        except Exception as e:
            logger.error(f"Summary streaming error: {e}")
            yield format_sse_event("error", {"message": str(e)})

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/stream/health")
async def stream_health() -> StreamingResponse:
    """
    Test endpoint for streaming functionality.

    Streams a simple health check message to verify SSE is working.
    """

    async def generate_events() -> AsyncIterator[str]:
        settings = get_settings()

        yield format_sse_event(
            "status",
            {
                "streaming": "operational",
                "llm_configured": settings.has_llm_key,
                "provider": settings.llm_provider.value,
            },
        )
        yield format_sse_event("done", {"message": "Streaming health check complete"})

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
