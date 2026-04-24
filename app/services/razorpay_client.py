"""Thin Razorpay API client for subscription flows."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class RazorpayAPIError(RuntimeError):
    """Raised when Razorpay returns a non-success response or a network error."""

    def __init__(self, message: str, *, status_code: int | None = None, payload: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


@dataclass(slots=True)
class RazorpayClient:
    """Minimal sync client used from background threads or low-volume endpoints."""

    key_id: str
    key_secret: str
    timeout_seconds: int = 15
    base_url: str = "https://api.razorpay.com/v1"

    def _request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(
                auth=(self.key_id, self.key_secret),
                timeout=self.timeout_seconds,
                headers={"Content-Type": "application/json"},
            ) as client:
                response = client.request(method, url, json=json_body)
        except httpx.HTTPError as exc:
            raise RazorpayAPIError("Razorpay request failed") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {"raw": response.text[:500]}

        if response.status_code >= 400:
            logger.warning("Razorpay API error %s on %s %s", response.status_code, method, path)
            raise RazorpayAPIError(
                "Razorpay API returned an error",
                status_code=response.status_code,
                payload=payload,
            )
        if not isinstance(payload, dict):
            raise RazorpayAPIError("Unexpected Razorpay response format", status_code=response.status_code, payload=payload)
        return payload

    def create_subscription(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a Razorpay subscription checkout object."""
        return self._request("POST", "/subscriptions", json_body=payload)

    def fetch_subscription(self, provider_subscription_id: str) -> dict[str, Any]:
        """Fetch one Razorpay subscription by id."""
        return self._request("GET", f"/subscriptions/{provider_subscription_id}")

    def cancel_subscription(self, provider_subscription_id: str, *, cancel_at_cycle_end: bool) -> dict[str, Any]:
        """Cancel a subscription now or at cycle end."""
        return self._request(
            "POST",
            f"/subscriptions/{provider_subscription_id}/cancel",
            json_body={"cancel_at_cycle_end": cancel_at_cycle_end},
        )


def get_razorpay_client() -> RazorpayClient:
    """Build a configured Razorpay client from application settings."""
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise RuntimeError("Razorpay is not configured")
    return RazorpayClient(
        key_id=settings.razorpay_key_id,
        key_secret=settings.razorpay_key_secret,
        timeout_seconds=settings.razorpay_timeout_seconds,
    )
