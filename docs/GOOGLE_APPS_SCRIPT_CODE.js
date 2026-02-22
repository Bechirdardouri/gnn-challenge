/**
 * Google Apps Script webhook bridge for Google Form -> GitHub Actions.
 *
 * 1) Update CONFIG values.
 * 2) Run setupGitHubPAT("ghp_...") once from Apps Script editor.
 * 3) Create trigger: onFormSubmit (From form / On form submit).
 * 4) Remove/comment setupGitHubPAT usage after configuration.
 */

const CONFIG = {
  owner: "YOUR_GITHUB_OWNER",
  repo: "YOUR_REPOSITORY",
  dispatchEventType: "google_form_submission",
};

function setupGitHubPAT(token) {
  if (!token || !token.startsWith("ghp_")) {
    throw new Error("Invalid PAT. Expected a classic token that starts with ghp_.");
  }
  PropertiesService.getScriptProperties().setProperty("GITHUB_PAT", token);
  return testGitHubPAT_();
}

function onFormSubmit(e) {
  return triggerGitHubDispatch_({
    event: "form_submit",
    triggerTime: new Date().toISOString(),
  });
}

function triggerGitHubDispatch_(payload) {
  const token = PropertiesService.getScriptProperties().getProperty("GITHUB_PAT");
  if (!token) {
    throw new Error("GITHUB_PAT not configured. Run setupGitHubPAT(token) first.");
  }

  const url = "https://api.github.com/repos/" + CONFIG.owner + "/" + CONFIG.repo + "/dispatches";
  const body = {
    event_type: CONFIG.dispatchEventType,
    client_payload: payload || {},
  };

  const response = UrlFetchApp.fetch(url, {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(body),
    muteHttpExceptions: true,
    headers: {
      Authorization: "Bearer " + token,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
    },
  });

  const code = response.getResponseCode();
  if (code !== 204) {
    throw new Error("GitHub dispatch failed. HTTP " + code + " - " + response.getContentText());
  }
  return "GitHub workflow dispatched successfully.";
}

function testGitHubPAT_() {
  const token = PropertiesService.getScriptProperties().getProperty("GITHUB_PAT");
  if (!token) {
    throw new Error("GITHUB_PAT not found in Script Properties.");
  }

  const url = "https://api.github.com/repos/" + CONFIG.owner + "/" + CONFIG.repo;
  const response = UrlFetchApp.fetch(url, {
    method: "get",
    muteHttpExceptions: true,
    headers: {
      Authorization: "Bearer " + token,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
    },
  });

  const code = response.getResponseCode();
  if (code !== 200) {
    throw new Error("PAT test failed. HTTP " + code + " - " + response.getContentText());
  }
  return "PAT test succeeded.";
}
