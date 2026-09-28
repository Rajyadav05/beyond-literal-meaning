/**
 * frontend/js/api.js
 * Shared API configuration and wrapper methods for Non-Literal Language Lab.
 */
const API_BASE_URL = "http://localhost:8000";

async function apiPost(endpoint, payload) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      let errMessage = `Error ${response.status}: Service request failed.`;
      try {
        const errJson = await response.json();
        if (errJson.error && errJson.error.message) {
          errMessage = errJson.error.message;
        } else if (errJson.detail) {
          errMessage = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
        } else if (typeof errJson.error === "string") {
          errMessage = errJson.error;
        }
      } catch (e) {
        // use default status error message
      }

      if (response.status === 503) {
        errMessage = errMessage || "Local NLP backend service is currently unavailable or initializing.";
      } else if (response.status === 422) {
        errMessage = errMessage || "Invalid input provided. Please enter a valid sentence.";
      }

      return { ok: false, status: response.status, error: errMessage };
    }

    const data = await response.json();
    return { ok: true, status: response.status, data: data };
  } catch (err) {
    return {
      ok: false,
      status: 0,
      error: "Backend unavailable. Please make sure the local NLP FastAPI backend (http://localhost:8000) is running."
    };
  }
}

async function apiGet(endpoint) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    if (!response.ok) {
      return { ok: false, status: response.status, error: `HTTP ${response.status}` };
    }
    const data = await response.json();
    return { ok: true, status: response.status, data: data };
  } catch (err) {
    return {
      ok: false,
      status: 0,
      error: "Backend unavailable. Please make sure the local NLP FastAPI backend (http://localhost:8000) is running."
    };
  }
}

// Export functions to global scope for static script imports
window.AppAPI = {
  baseUrl: API_BASE_URL,
  checkHealth: () => apiGet("/api/health"),
  analyze: (sentence, context) => {
    const body = { sentence: sentence };
    if (context && context.trim()) {
      body.context = context.trim();
    }
    return apiPost("/api/analyze", body);
  },
  compareContext: (sentence, context) => {
    return apiPost("/api/analyze/context", { sentence: sentence, context: context });
  },
  analyzeIdiom: (sentence, context) => {
    const body = { sentence: sentence };
    if (context && context.trim()) {
      body.context = context.trim();
    }
    return apiPost("/api/idiom", body);
  },
  submitChallenge: (payload) => {
    return apiPost("/api/challenge", payload);
  },
  getChallengeStats: () => {
    return apiGet("/api/challenge/stats");
  }
};

