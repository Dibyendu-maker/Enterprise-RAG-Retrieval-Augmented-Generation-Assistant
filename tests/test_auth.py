import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_user_registration_success(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "john.doe@enterprise.com",
            "password": "Password123!",
            "full_name": "John Doe"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "john.doe@enterprise.com"
    assert data["full_name"] == "John Doe"
    assert "id" in data
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_duplicate_registration_fails(client: AsyncClient):
    payload = {
        "email": "duplicate@enterprise.com",
        "password": "Password123!",
        "full_name": "Duplicate User"
    }
    res1 = await client.post("/api/v1/auth/register", json=payload)
    assert res1.status_code == 201

    res2 = await client.post("/api/v1/auth/register", json=payload)
    assert res2.status_code == 400
    assert "already exists" in res2.json()["detail"]


@pytest.mark.asyncio
async def test_login_oauth2_and_json(client: AsyncClient):
    # Register user
    await client.post(
        "/api/v1/auth/register",
        json={"email": "loginuser@enterprise.com", "password": "MySecretPassword1!"}
    )

    # OAuth2 form login
    oauth_res = await client.post(
        "/api/v1/auth/login",
        data={"username": "loginuser@enterprise.com", "password": "MySecretPassword1!"}
    )
    assert oauth_res.status_code == 200
    assert "access_token" in oauth_res.json()
    assert oauth_res.json()["token_type"] == "bearer"

    # JSON login
    json_res = await client.post(
        "/api/v1/auth/login-json",
        json={"email": "loginuser@enterprise.com", "password": "MySecretPassword1!"}
    )
    assert json_res.status_code == 200
    assert "access_token" in json_res.json()


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "wrongpwd@enterprise.com", "password": "CorrectPassword1!"}
    )
    res = await client.post(
        "/api/v1/auth/login-json",
        json={"email": "wrongpwd@enterprise.com", "password": "IncorrectPassword!"}
    )
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_auth_me_endpoint(client: AsyncClient, auth_headers: dict):
    response = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "alice@enterprise.com"
    assert data["full_name"] == "Alice Developer"


@pytest.mark.asyncio
async def test_auth_me_unauthorized(client: AsyncClient):
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401
