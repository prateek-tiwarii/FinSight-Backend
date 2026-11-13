from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.auth import UserPublic
from tests.helpers import (
    csrf_headers,
    make_guest_client,
    prime_csrf,
    register_client,
)


def _new_client() -> TestClient:
    client = TestClient(app)
    prime_csrf(client)
    return client


def test_me_issues_csrf_cookie_and_reports_anonymous():
    client = TestClient(app)
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    assert response.json()["authenticated"] is False
    assert client.cookies.get(settings.csrf_cookie_name)


def test_unsafe_requests_require_csrf_header():
    client = TestClient(app)
    client.get("/api/auth/me")
    response = client.post(
        "/api/guest/session",
        headers={"X-CSRF-Token": "wrong-token"},
    )
    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


def test_register_login_logout_and_duplicate_email():
    client = _new_client()
    email = f"auth-{uuid.uuid4().hex[:8]}@example.com"
    payload = {
        "full_name": "Auth Tester",
        "email": email,
        "password": "Str0ng!Passw0rd",
        "confirm_password": "Str0ng!Passw0rd",
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201, response.text
    body = UserPublic.model_validate(response.json()["user"])
    assert body.email == email
    assert settings.session_cookie_name in response.cookies

    duplicate = client.post("/api/auth/register", json=payload)
    assert duplicate.status_code == 409

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert client.get("/api/auth/me").json()["authenticated"] is False

    wrong = client.post(
        "/api/auth/login", json={"email": email, "password": "WrongPass1!"}
    )
    assert wrong.status_code == 401

    login = client.post(
        "/api/auth/login", json={"email": email, "password": "Str0ng!Passw0rd"}
    )
    assert login.status_code == 200
    assert client.get("/api/auth/me").json()["user"]["email"] == email


def test_register_rejects_weak_or_mismatched_passwords():
    client = _new_client()
    base = {
        "full_name": "Weak Tester",
        "email": f"weak-{uuid.uuid4().hex[:8]}@example.com",
    }
    weak = client.post(
        "/api/auth/register",
        json={**base, "password": "password", "confirm_password": "password"},
    )
    assert weak.status_code == 422
    mismatch = client.post(
        "/api/auth/register",
        json={
            **base,
            "password": "Str0ng!Passw0rd",
            "confirm_password": "Str0ng!Passw0rd2",
        },
    )
    assert mismatch.status_code == 422


def test_guest_session_is_idempotent_and_cookie_is_http_only():
    client = _new_client()
    first = client.post("/api/guest/session")
    assert first.status_code == 201
    first_guest = first.json()["guest"]["id"]
    second = client.post("/api/guest/session")
    assert second.status_code in {200, 201}
    assert second.json()["guest"]["id"] == first_guest

    cookie_header = first.headers.get("set-cookie", "")
    assert "HttpOnly" in cookie_header


def test_guest_data_is_isolated_between_guests():
    alice = make_guest_client()
    bob = make_guest_client()
    created = alice.post("/api/portfolios", json={"name": "Alice Only"})
    assert created.status_code == 201, created.text
    portfolio_id = created.json()["id"]

    assert bob.get(f"/api/portfolios/{portfolio_id}").status_code == 404
    assert bob.get("/api/portfolios").json()["portfolios"] == []
    assert alice.get(f"/api/portfolios/{portfolio_id}").status_code == 200


def test_guest_limits_are_enforced(monkeypatch):
    monkeypatch.setattr(settings, "guest_max_portfolios", 1)
    client = make_guest_client()
    assert client.post("/api/portfolios", json={"name": "One"}).status_code == 201
    blocked = client.post("/api/portfolios", json={"name": "Two"})
    assert blocked.status_code == 403
    assert "limit" in blocked.json()["detail"].lower()


def test_protected_routes_require_principal():
    client = _new_client()
    response = client.post("/api/portfolios", json={"name": "Anonymous"})
    assert response.status_code == 401


def test_guest_conversion_requires_confirmation_and_moves_ownership():
    guest = make_guest_client()
    created = guest.post("/api/portfolios", json={"name": "Guest Portfolio"})
    portfolio_id = created.json()["id"]

    email = f"convert-{uuid.uuid4().hex[:8]}@example.com"
    registered = register_client(guest, email=email)
    # register_client returns the client; confirm the conversion offer via /auth/me.
    me = registered.get("/api/auth/me").json()
    assert me["authenticated"] is True
    assert me["guest_conversion_available"] is True

    no_confirm = registered.post("/api/guest/convert", json={"confirm_merge": False})
    assert no_confirm.status_code == 409
    assert no_confirm.json()["detail"]["requires_confirmation"] is True

    converted = registered.post("/api/guest/convert", json={"confirm_merge": True})
    assert converted.status_code == 200, converted.text
    assert converted.json()["converted"] is True
    assert converted.json()["counts"]["portfolios"] >= 1

    portfolios = registered.get("/api/portfolios").json()["portfolios"]
    assert any(p["id"] == portfolio_id for p in portfolios)
    assert registered.get("/api/guest/me").json()["authenticated"] is False
    assert registered.post("/api/guest/convert", json={"confirm_merge": True}).status_code == 400


def test_registered_user_cannot_read_another_guests_portfolio():
    guest = make_guest_client()
    created = guest.post("/api/portfolios", json={"name": "Guest Secret"})
    portfolio_id = created.json()["id"]
    user = register_client()
    assert user.get(f"/api/portfolios/{portfolio_id}").status_code == 404


def test_csrf_header_helper_matches_cookie():
    client = make_guest_client()
    assert csrf_headers(client)["X-CSRF-Token"] == client.cookies.get(
        settings.csrf_cookie_name
    )
