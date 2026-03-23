# Frontend Authentication Integration Guide

This guide explains how to integrate the AI English Practice backend authentication into your frontend application. The backend uses **OAuth on the client** (Google and Apple Sign-In); the frontend obtains an **ID token** from the provider and exchanges it for **backend-issued JWT access and refresh tokens**.

---

## 1. Overview

```text
┌─────────────┐     ID token      ┌─────────────┐     JWT pair + user     ┌─────────────┐
│   Flutter   │ ────────────────► │   Backend   │ ──────────────────────► │   Flutter   │
│  (OAuth UI) │                    │  /auth/oauth │                        │ (store + use)│
└─────────────┘                    └─────────────┘                         └─────────────┘
       │                                   │
       │  Google / Apple Sign-In SDK        │  Verify ID token, create user,
       │  returns id_token                  │  issue access + refresh JWT
       └───────────────────────────────────┘
```

- **OAuth** (Google / Apple) runs entirely on the client. Your app uses the provider’s SDK to sign in and receive an **ID token**.
- The backend does **not** redirect users to Google/Apple. You send the ID token to the backend once per sign-in.
- The backend verifies the ID token, creates or finds the user, and returns:
  - **Access token** (JWT, short-lived, e.g. 15 minutes)
  - **Refresh token** (JWT, long-lived, e.g. 30 days)
  - **User** (id, email, name, **onboarding_completed**, **onboarding_step**)
- **After login**, users must complete a **five-step onboarding flow** before they can use AI features (chat, voice, TTS). The `user` object in `/oauth` and `/me` includes `onboarding_completed` and `onboarding_step` so the client can show the onboarding screens or the main app.
- All subsequent API calls use the **access token** in the `Authorization` header. When the access token expires, use the **refresh token** to get a new access token.

---

## 2. Base URL and Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/auth/oauth` | Exchange Google/Apple ID token for JWTs and user |
| `POST` | `/api/v1/auth/refresh` | Get a new access token using refresh token |
| `GET`  | `/api/v1/auth/me`    | Get current user (requires access token) |
| `POST` | `/api/v1/auth/logout` | Log out (requires access token); client must clear tokens |
| `GET`  | `/api/v1/auth/onboarding/status` | Get onboarding progress (requires access token) |
| `POST` | `/api/v1/auth/onboarding/nickname` | Step 1: set nickname |
| `POST` | `/api/v1/auth/onboarding/language` | Step 2: set native language |
| `POST` | `/api/v1/auth/onboarding/student` | Step 3: set student type and occupation |
| `POST` | `/api/v1/auth/onboarding/goal` | Step 4: set goal |
| `POST` | `/api/v1/auth/onboarding/english-level` | Step 5: set English level (completes onboarding) |

**Base URL:** Your backend root (e.g. `https://api.example.com` or `http://localhost:8000`).

**Rate limits:** Auth endpoints are limited to **10 requests per minute per IP**. Return and display a user-friendly message when you receive `429 Too Many Requests`.

---

## 3. Request and Response Contracts

### 3.1 Login: `POST /api/v1/auth/oauth`

**Request body (JSON):**

```json
{
  "provider": "google",
  "id_token": "<id_token_from_google_or_apple_sdk>"
}
```

- `provider`: `"google"` or `"apple"`.
- `id_token`: The ID token string returned by the provider’s SDK after sign-in.

**Success response (200):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com",
    "name": "Jane Doe",
    "onboarding_completed": false,
    "onboarding_step": 0
  }
}
```

**Error responses:**

- `401 Unauthorized` – Invalid or expired ID token. Body: `{ "detail": "<message>" }`.
- `422 Unprocessable Entity` – Validation error (e.g. missing `provider` or `id_token`).
- `429 Too Many Requests` – Rate limit exceeded.

---

### 3.2 Refresh: `POST /api/v1/auth/refresh`

**Request body (JSON):**

```json
{
  "refresh_token": "<stored_refresh_token>"
}
```

**Success response (200):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Error responses:**

- `401 Unauthorized` – Invalid or expired refresh token. Body: `{ "detail": "Invalid or expired refresh token" }`.
- `429 Too Many Requests` – Rate limit exceeded.

---

### 3.3 Current user: `GET /api/v1/auth/me`

**Headers:**

```http
Authorization: Bearer <access_token>
```

**Success response (200):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "name": "Jane Doe",
  "onboarding_completed": false,
  "onboarding_step": 2
}
```

**Error responses:**

- `401 Unauthorized` – Missing, invalid, or expired access token. Do not send a body for `Authorization` missing; for invalid/expired token the body is `{ "detail": "..." }`.
- `429 Too Many Requests` – Rate limit exceeded.

---

### 3.4 Logout: `POST /api/v1/auth/logout`

Call this when the user taps “Log out”. The backend does **not** invalidate JWTs (tokens are stateless); this endpoint confirms the user was authenticated and signals that the client should clear all stored tokens.

**Headers:**

```http
Authorization: Bearer <access_token>
```

**Success response (200):**

```json
{
  "message": "Successfully logged out"
}
```

**Client action after 200:** Clear the access token and refresh token from memory and secure storage, clear any in-app user state, and redirect to the login screen.

**Error responses:**

- `401 Unauthorized` – Missing or invalid access token. You can still clear tokens locally and redirect to login.
- `429 Too Many Requests` – Rate limit exceeded.

---

## 3.5 Onboarding

All onboarding endpoints require `Authorization: Bearer <access_token>`. After login or `GET /me`, use `user.onboarding_completed` and `user.onboarding_step` to decide whether to show onboarding (steps 1–5) or the main app. When `onboarding_step` is 5, `onboarding_completed` is `true` and the user can access AI features.

### GET /api/v1/auth/onboarding/status

Returns current progress. No request body.

**Success response (200):**

```json
{
  "onboarding_completed": false,
  "current_step": 2
}
```

### POST /api/v1/auth/onboarding/nickname (Step 1)

**Request body:** `{ "nickname": "Sahal" }` — nickname must be 3–30 characters.

**Success (200):** `{ "onboarding_step": 1 }`

### POST /api/v1/auth/onboarding/language (Step 2)

**Request body:** `{ "native_language": "Malayalam" }` — required, 1–100 characters.

**Success (200):** `{ "onboarding_step": 2 }`

### POST /api/v1/auth/onboarding/student (Step 3)

**Request body:**

```json
{
  "student_type": "adult",
  "occupation": "college"
}
```

- `student_type`: `"adult"` or `"kid"`
- `occupation`: `"college"` | `"work"` | `"home_maker"` | `"teacher"` | `"other"`

**Success (200):** `{ "onboarding_step": 3 }`

### POST /api/v1/auth/onboarding/goal (Step 4)

**Request body:** `{ "goal": "prepare_interviews" }`

- `goal`: `"prepare_interviews"` | `"prepare_govt_exams"` | `"go_abroad"` | `"talk_family"` | `"build_confidence"` | `"improve_workplace"` | `"other"`

**Success (200):** `{ "onboarding_step": 4 }`

### POST /api/v1/auth/onboarding/english-level (Step 5)

**Request body:** `{ "english_level": "beginner" }`

- `english_level`: `"beginner"` | `"elementary"` | `"intermediate"` | `"upper_intermediate"` | `"advanced"`

**Success (200):** `{ "onboarding_step": 5, "onboarding_completed": true }`

After this, the user can call AI endpoints. Call `GET /me` again to refresh the stored user object with `onboarding_completed: true` if needed.

**Error responses (all onboarding endpoints):** 401 (missing/invalid token), 422 (validation error), 429 (rate limit).

---

## 4. Token Storage and Security

- **Access token:** Prefer in-memory only (e.g. variable in app state / service). If you must persist (e.g. for cold start), use a short-lived store and treat it as sensitive.
- **Refresh token:** Store securely:
  - **Flutter:** e.g. `flutter_secure_storage` (encrypted, not in plain SharedPreferences).
  - **Web:** Avoid `localStorage` for refresh token if possible; consider httpOnly cookies if the backend supports it, or a secure in-memory + optional encrypted persistence.
- Do not log or send tokens to analytics. Do not put tokens in URLs or query params.

---

## 5. Sending the Access Token on API Requests

For every request to a **protected** backend route (e.g. `/api/v1/auth/me`, `/api/v1/auth/logout`, or any future user-scoped API), send the current access token:

```http
Authorization: Bearer <access_token>
```

Example header in Dart/Flutter:

```dart
headers: {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer $accessToken',
}
```

Use the **access token** only. Do not send the refresh token for normal API calls.

---

## 6. Refresh Flow (When Access Token Expires)

- Access tokens are short-lived (e.g. **15 minutes**). When a request returns **401** with a message indicating invalid/expired token:
  1. Call `POST /api/v1/auth/refresh` with the stored **refresh_token** in the body.
  2. If refresh returns **200**, replace the stored access token with the new `access_token` and retry the original request.
  3. If refresh returns **401**, treat the session as ended: clear tokens and user, and redirect to login.

**Recommended:** Implement a single place (e.g. HTTP client interceptor or API service) that:

- Adds `Authorization: Bearer <access_token>` to outgoing requests.
- On **401** from any protected endpoint, tries refresh once, then retries the request; if refresh fails, logs out (clear tokens; optionally call `POST /api/v1/auth/logout` if you still have a valid access token), then redirect to login.

---

## 7. Error Handling Summary

| Status | Meaning | Suggested action |
|--------|---------|------------------|
| 200    | Success | Use response body; after login/refresh, store tokens and user. |
| 401    | Unauthorized (invalid/expired token or invalid OAuth token) | On login/refresh: show error. On API call: try refresh once, then logout. |
| 403    | Forbidden – onboarding not completed | On AI endpoints: redirect user to onboarding; do not retry. |
| 422    | Validation error | Show field or message from `detail`. |
| 429    | Rate limit | Show “Too many attempts. Try again in a minute.” |
| 5xx    | Server error | Show generic “Something went wrong” and optionally retry. |

Parse `detail` from the JSON body when the backend returns it (e.g. 401, 422) and show it to the user where appropriate.

---

## 8. Flutter-Oriented Flow

### 8.1 Dependencies (example)

- Google Sign-In: `google_sign_in`
- Apple Sign-In: `sign_in_with_apple`
- Secure storage: `flutter_secure_storage`
- HTTP: `dio` or `http`

### 8.2 High-Level Login Flow

1. User taps “Sign in with Google” or “Sign in with Apple”.
2. Use the provider plugin to perform sign-in and obtain the **ID token** (and optionally user profile).
3. Call your backend:

```dart
// Pseudocode
final idToken = await getGoogleOrAppleIdToken(); // from plugin
final response = await http.post(
  Uri.parse('$baseUrl/api/v1/auth/oauth'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'provider': 'google', // or 'apple'
    'id_token': idToken,
  }),
);

if (response.statusCode == 200) {
  final data = jsonDecode(response.body);
  final accessToken = data['access_token'];
  final refreshToken = data['refresh_token'];
  final user = data['user'];
  final onboardingCompleted = user['onboarding_completed'] ?? false;
  final onboardingStep = user['onboarding_step'] ?? 0;
  await secureStorage.write(key: 'refresh_token', value: refreshToken);
  // Store accessToken in memory or short-lived storage; set current user.
  // If !onboardingCompleted, show onboarding from step (onboardingStep + 1).
} else if (response.statusCode == 401) {
  // Show: Invalid sign-in. Try again.
} else if (response.statusCode == 429) {
  // Show: Too many attempts.
}
```

4. For subsequent requests, set `Authorization: Bearer $accessToken`. When you get 401, call refresh (see §6), then retry.

### 8.3 Logout flow

When the user taps “Log out”:

1. (Optional) Call `POST /api/v1/auth/logout` with `Authorization: Bearer <access_token>`. If the token is already expired, skip or ignore 401.
2. Clear the access token and refresh token from memory and secure storage (e.g. `flutter_secure_storage.delete(key: 'refresh_token')`).
3. Clear in-app user state and navigate to the login screen.

```dart
// Pseudocode
try {
  await http.post(
    Uri.parse('$baseUrl/api/v1/auth/logout'),
    headers: {'Authorization': 'Bearer $accessToken'},
  );
} catch (_) {}
await secureStorage.delete(key: 'refresh_token');
// Clear accessToken and user from state; go to login.
```

### 8.4 Apple Sign-In Notes

- Apple may return the user’s email and name only on the **first** successful sign-in. The backend stores them; later sign-ins may not include name/email in the Apple token, but `/auth/me` will still return the stored user.
- Ensure your Apple App ID and Services (Sign in with Apple) are configured in Apple Developer and that the backend `APPLE_CLIENT_ID` matches (e.g. app bundle ID or services ID).

### 8.5 Google Sign-In Notes

- Use the **server-side client ID** or the correct audience when requesting the ID token so the backend (which uses `GOOGLE_CLIENT_ID`) can verify it.
- Request scopes needed for email/name if you want them in the token; the backend reads `email` and `name` from the verified token.

---

## 9. AI Endpoints and Onboarding

All AI endpoints (e.g. `/api/v1/ai/text-chat`, `/api/v1/ai/voice-chat`, `/api/v1/ai/chat/stream`, `/api/v1/ai/voice-chat/stream`, `/api/v1/ai/tts/stream`, `/api/v1/ai/init-models`) **require authentication** (`Authorization: Bearer <access_token>`) and **completed onboarding**. If the user has not finished onboarding (`onboarding_completed === false`), the backend returns **403 Forbidden** with `detail: "User onboarding not completed"`. The client should:

- After login or `GET /me`, branch on `user.onboarding_completed`. If `false`, show the onboarding flow and call the onboarding step endpoints in order (1–5).
- When calling any AI endpoint, if the response is **403** with that detail, redirect the user to the onboarding screen (or the step indicated by `user.onboarding_step`) instead of retrying.
- Use `user.id` from the auth response as the authenticated user; the backend uses the JWT to resolve the user and no longer relies on a `user_id` in the request body for authorization.

---

## 10. Quick Checklist

- [ ] Implement Google and/or Apple Sign-In and obtain the **ID token**.
- [ ] Call `POST /api/v1/auth/oauth` with `provider` and `id_token`; handle 200, 401, 422, 429.
- [ ] Store **refresh_token** securely; keep **access_token** in memory (or short-lived secure storage).
- [ ] Send `Authorization: Bearer <access_token>` on every request to protected endpoints.
- [ ] On 401 from a protected endpoint, call `POST /api/v1/auth/refresh` with `refresh_token`, then retry; if refresh fails, clear session and show login.
- [ ] Use `GET /api/v1/auth/me` to restore user profile on app start when you have a valid access token (or after refreshing).
- [ ] After login or `GET /me`, check `user.onboarding_completed`. If `false`, show onboarding and call the onboarding step endpoints in order (nickname → language → student → goal → english-level).
- [ ] When `onboarding_step` reaches 5 (or `onboarding_completed` is `true`), allow access to AI features; optionally refresh user with `GET /me`.
- [ ] On **403** from any AI endpoint with `detail: "User onboarding not completed"`, redirect to onboarding (or the current step) instead of retrying.
- [ ] On logout: call `POST /api/v1/auth/logout` (optional, with Bearer token), then clear access token, refresh token, and user state; redirect to login.
- [ ] Handle 429 with a user-friendly “too many attempts” message.
- [ ] Never log or expose tokens; do not send refresh token in the `Authorization` header.
