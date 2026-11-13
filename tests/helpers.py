from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def csrf_headers(client: TestClient) -> dict[str, str]:
    token = client.cookies.get(settings.csrf_cookie_name)
    return {"X-CSRF-Token": token} if token else {}


def prime_csrf(client: TestClient) -> None:
    client.get("/api/auth/me")
    token = client.cookies.get(settings.csrf_cookie_name)
    assert token
    # Double-submit: mirror the readable cookie on every unsafe request.
    client.headers.update({"X-CSRF-Token": token})


def make_guest_client() -> TestClient:
    client = TestClient(app)
    prime_csrf(client)
    response = client.post("/api/guest/session", headers=csrf_headers(client))
    assert response.status_code in {200, 201}, response.text
    return client


def register_client(
    client: TestClient | None = None,
    *,
    email: str | None = None,
    password: str = "Str0ng!Passw0rd",
    full_name: str = "Test User",
) -> TestClient:
    client = client or TestClient(app)
    prime_csrf(client)
    address = email or f"user-{uuid.uuid4().hex[:10]}@example.com"
    response = client.post(
        "/api/auth/register",
        json={
            "full_name": full_name,
            "email": address,
            "password": password,
            "confirm_password": password,
        },
        headers=csrf_headers(client),
    )
    assert response.status_code == 201, response.text
    return client
