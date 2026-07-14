(function () {
    "use strict";

    var pageEl = document.querySelector(".open-source-work");
    var DEFAULT_REPOS = [
        "intel/intel-xpu-backend-for-triton",
        "intel/torch-xpu-ops",
        "pytorch/pytorch",
        "vllm-project/vllm"
    ];
    var API_ROOT = "https://api.github.com/repos";
    var PER_PAGE = 100;

    function readConfiguredRepos() {
        var configured = [];
        var repoPattern = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

        if (pageEl) {
            try {
                configured = JSON.parse(pageEl.getAttribute("data-github-repos") || "[]");
            } catch (error) {
                configured = [];
            }
        }

        configured = Array.isArray(configured) ? configured : [];
        configured = configured.map(function (repo) {
            return String(repo || "").trim();
        }).filter(function (repo, index, repos) {
            return repoPattern.test(repo) && repos.indexOf(repo) === index;
        });

        return configured.length ? configured : DEFAULT_REPOS.slice();
    }

    var USER = pageEl && pageEl.getAttribute("data-github-user") ? pageEl.getAttribute("data-github-user") : "sgurwinderr";
    var REPOS = readConfiguredRepos();
    var CACHE_KEY = "osw-live-results-v9-" + USER + "-" + REPOS.join("|");

    var resultsEl = document.getElementById("osw-results");
    var statusEl = document.getElementById("osw-status");
    var refreshBtn = document.getElementById("osw-refresh");
    var filterEls = Array.prototype.slice.call(document.querySelectorAll(".osw-filter-btn"));
    var filterCountEl = document.getElementById("osw-filter-count");
    var countEls = {
        pr: document.getElementById("osw-pr-count"),
        closedPr: document.getElementById("osw-closed-pr-count"),
        issue: document.getElementById("osw-issue-count"),
        closedIssue: document.getElementById("osw-closed-issue-count"),
        authored: document.getElementById("osw-authored-count"),
        assigned: document.getElementById("osw-assigned-count"),
        filterAll: document.getElementById("osw-filter-all-count"),
        filterIssues: document.getElementById("osw-filter-issues-count"),
        filterOpenPr: document.getElementById("osw-filter-open-pr-count"),
        filterClosedPr: document.getElementById("osw-filter-closed-pr-count"),
        filterAuthored: document.getElementById("osw-filter-authored-count")
    };
    var activeFilter = "all";
    var allItems = [];

    if (!resultsEl || !statusEl) {
        return;
    }

    function escapeHtml(value) {
        return String(value == null ? "" : value).replace(/[&<>"']/g, function (char) {
            return {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                "\"": "&quot;",
                "'": "&#39;"
            }[char];
        });
    }

    function setStatus(message, kind) {
        statusEl.textContent = message;
        statusEl.className = "osw-status" + (kind ? " osw-status-" + kind : "");
        statusEl.hidden = false;
    }

    function hideStatus() {
        statusEl.hidden = true;
    }

    function buildUrl(repo, qualifier) {
        var params = new URLSearchParams({
            state: "all",
            sort: "updated",
            direction: "desc",
            per_page: String(PER_PAGE)
        });
        params.set(qualifier === "creator" ? "creator" : "assignee", USER);
        return API_ROOT + "/" + repo + "/issues?" + params.toString();
    }

    function fetchQuery(repo, qualifier) {
        return fetch(buildUrl(repo, qualifier), {
            headers: { Accept: "application/vnd.github+json" }
        }).then(function (response) {
            if (!response.ok) {
                return response.json()["catch"](function () {
                    return {};
                }).then(function (body) {
                    var error = new Error(body.message || "GitHub API request failed.");
                    error.status = response.status;
                    throw error;
                });
            }
            return response.json();
        }).then(function (items) {
            return (items || []).map(function (item) {
                return {
                    raw: item,
                    repo: repo,
                    role: qualifier === "creator" ? "Author" : "Assigned"
                };
            });
        });
    }

    function fetchSequentially(queries) {
        var responses = [];
        return queries.reduce(function (chain, query) {
            return chain.then(function () {
                return query().then(function (items) {
                    responses.push(items);
                });
            });
        }, Promise.resolve()).then(function () {
            return responses;
        });
    }

    function readCachedItems() {
        try {
            var cached = JSON.parse(localStorage.getItem(CACHE_KEY) || "null");
            if (cached && Array.isArray(cached.items)) {
                return cached;
            }
        } catch (error) {
            localStorage.removeItem(CACHE_KEY);
        }
        return null;
    }

    function writeCachedItems(items) {
        try {
            localStorage.setItem(CACHE_KEY, JSON.stringify({
                savedAt: new Date().toISOString(),
                items: items
            }));
        } catch (error) {
            // Storage is best effort only.
        }
    }

    function normalizeItems(responses) {
        var byKey = new Map();

        responses.flat().forEach(function (entry) {
            var item = entry.raw;
            var isPullRequest = Boolean(item.pull_request);
            var key = item.repository_url + "#" + item.number;
            var existing = byKey.get(key);

            if (!existing) {
                existing = {
                    id: item.id,
                    repo: entry.repo,
                    kind: isPullRequest && item.state === "closed" ? "closedPr" : isPullRequest ? "pr" : item.state === "closed" ? "closedIssue" : "issue",
                    number: item.number,
                    title: item.title,
                    url: item.html_url,
                    updatedAt: item.updated_at,
                    labels: item.labels || [],
                    comments: item.comments || 0,
                    roles: []
                };
                byKey.set(key, existing);
            }

            if (existing.roles.indexOf(entry.role) === -1) {
                existing.roles.push(entry.role);
            }
        });

        return Array.from(byKey.values()).sort(function (a, b) {
            return new Date(b.updatedAt) - new Date(a.updatedAt);
        });
    }

    function countItems(items) {
        return items.reduce(function (acc, item) {
            acc[item.kind] += 1;
            if (item.roles.indexOf("Author") !== -1) {
                acc.authored += 1;
            }
            if (item.roles.indexOf("Assigned") !== -1) {
                acc.assigned += 1;
            }
            return acc;
        }, { pr: 0, closedPr: 0, issue: 0, closedIssue: 0, authored: 0, assigned: 0 });
    }

    function setText(el, value) {
        if (el) {
            el.textContent = String(value);
        }
    }

    function setCounts(items) {
        var counts = countItems(items);
        setText(countEls.pr, counts.pr);
        setText(countEls.closedPr, counts.closedPr);
        setText(countEls.issue, counts.issue);
        setText(countEls.closedIssue, counts.closedIssue);
        setText(countEls.authored, counts.authored);
        setText(countEls.assigned, counts.assigned);
        setText(countEls.filterAll, items.length);
        setText(countEls.filterIssues, counts.issue + counts.closedIssue);
        setText(countEls.filterOpenPr, counts.pr);
        setText(countEls.filterClosedPr, counts.closedPr);
        setText(countEls.filterAuthored, counts.authored);
    }

    function hasActiveFilters() {
        return activeFilter !== "all";
    }

    function itemMatchesFilters(item) {
        if (activeFilter === "all") {
            return true;
        }
        if (activeFilter === "issues") {
            return item.kind === "issue" || item.kind === "closedIssue";
        }
        if (activeFilter === "open-pr") {
            return item.kind === "pr";
        }
        if (activeFilter === "closed-pr") {
            return item.kind === "closedPr";
        }
        if (activeFilter === "authored") {
            return item.roles.indexOf("Author") !== -1;
        }
        return true;
    }

    function getVisibleItems() {
        return allItems.filter(itemMatchesFilters);
    }

    function updateFilterButtons() {
        filterEls.forEach(function (button) {
            var key = button.getAttribute("data-filter");
            var active = key === activeFilter;
            button.classList.toggle("is-active", active);
            button.setAttribute("aria-selected", active ? "true" : "false");
            button.setAttribute("tabindex", active ? "0" : "-1");
        });
    }

    function updateFilterCount(visibleCount, totalCount) {
        if (!filterCountEl) {
            return;
        }
        if (!totalCount) {
            filterCountEl.textContent = "";
            return;
        }
        filterCountEl.textContent = hasActiveFilters() ? visibleCount + " of " + totalCount + " shown" : "All " + totalCount + " shown";
    }

    function formatDate(value) {
        return new Intl.DateTimeFormat(undefined, {
            month: "short",
            day: "numeric",
            year: "numeric"
        }).format(new Date(value));
    }

    function latestUpdatedText(items) {
        if (!items.length) {
            return "No matching activity";
        }
        return "Updated " + formatDate(items[0].updatedAt);
    }

    function pluralize(count, singular, plural) {
        return count + " " + (count === 1 ? singular : plural);
    }

    function stateText(kind) {
        if (kind === "pr") return "Open PR";
        if (kind === "closedPr") return "Closed PR";
        if (kind === "closedIssue") return "Closed Issue";
        return "Open Issue";
    }

    function stateClass(kind) {
        if (kind === "closedPr") return "closed-pr";
        if (kind === "closedIssue") return "closed-issue";
        return kind;
    }

    function renderCompactRoles(roles) {
        return roles.map(escapeHtml).join(" / ");
    }

    function renderRow(item) {
        return [
            '<a class="osw-row" href="' + escapeHtml(item.url) + '" target="_blank" rel="noopener noreferrer">',
                '<div class="osw-row-main">',
                    '<div class="osw-row-titleline">',
                        '<span class="osw-state osw-state-' + stateClass(item.kind) + '">' + stateText(item.kind) + "</span>",
                        '<span class="osw-row-title">' + escapeHtml(item.title) + "</span>",
                    "</div>",
                    '<div class="osw-row-meta">',
                        '<span>#' + item.number + "</span>",
                        '<span class="osw-dot">' + formatDate(item.updatedAt) + "</span>",
                        '<span class="osw-dot">' + renderCompactRoles(item.roles) + "</span>",
                    "</div>",
                "</div>",
                '<span class="osw-external">GitHub</span>',
            "</a>"
        ].join("");
    }

    function renderRepo(repo, items, repoIndex) {
        var counts = countItems(items);
        var expanded = expandedRepos[repo] === true;
        var panelId = repoPanelId(repo);
        var body = items.length ? '<div class="osw-work-list">' + items.map(renderRow).join("") + "</div>" : '<div class="osw-empty-repo">No matching items in this repository.</div>';
        return [
            '<article class="osw-repo' + (expanded ? " is-open" : "") + '" style="animation-delay:' + (repoIndex * 70) + 'ms">',
                '<div class="osw-repo-shell" data-repo-shell>',
                    '<button class="osw-repo-header" type="button" data-repo-toggle="' + escapeHtml(repo) + '" aria-expanded="' + (expanded ? "true" : "false") + '" aria-controls="' + panelId + '">',
                        '<span class="osw-repo-heading">',
                            '<span class="osw-repo-eyeline">' + pluralize(items.length, "item", "items") + "</span>",
                            '<span class="osw-repo-title">' + escapeHtml(repo) + "</span>",
                            '<span class="osw-repo-subtitle">' + latestUpdatedText(items) + "</span>",
                        "</span>",
                        '<span class="osw-repo-actions">',
                            '<span class="osw-repo-counts" aria-label="Repository counts">',
                                '<span class="osw-count-chip osw-count-open">' + (counts.pr + counts.closedPr) + " PRs</span>",
                                '<span class="osw-count-chip osw-count-issues">' + (counts.issue + counts.closedIssue) + " Issues</span>",
                            "</span>",
                            '<span class="osw-repo-chevron" aria-hidden="true"></span>',
                        "</span>",
                    "</button>",
                    '<div class="osw-repo-panel" id="' + panelId + '"' + (expanded ? "" : " hidden") + ">",
                        body,
                    "</div>",
                "</div>",
            "</article>"
        ].join("");
    }

    function renderSkeleton() {
        resultsEl.innerHTML = [
            '<div class="osw-skeleton-list" aria-hidden="true">',
                '<div class="osw-skeleton-row"><div class="osw-skeleton-line short"></div><div class="osw-skeleton-line long"></div><div class="osw-skeleton-line mid"></div></div>',
                '<div class="osw-skeleton-row"><div class="osw-skeleton-line short"></div><div class="osw-skeleton-line long"></div><div class="osw-skeleton-line mid"></div></div>',
                '<div class="osw-skeleton-row"><div class="osw-skeleton-line short"></div><div class="osw-skeleton-line long"></div><div class="osw-skeleton-line mid"></div></div>',
            "</div>"
        ].join("");
    }

    function render(items) {
        allItems = items || [];
        setCounts(allItems);
        updateFilterButtons();

        var visibleItems = getVisibleItems();
        updateFilterCount(visibleItems.length, allItems.length);

        if (!allItems.length) {
            resultsEl.innerHTML = "";
            setStatus("No PRs or issues matched the tracked repositories.", "empty");
            return;
        }

        resultsEl.innerHTML = REPOS.map(function (repo, index) {
            return renderRepo(repo, visibleItems.filter(function (item) {
                return item.repo === repo;
            }), index);
        }).join("");

        if (!visibleItems.length) {
            setStatus("No items match the selected filters.", "empty");
        } else {
            hideStatus();
        }
    }

    function loadResults() {
        var queries = [];
        REPOS.forEach(function (repo) {
            ["creator", "assignee"].forEach(function (qualifier) {
                queries.push(function () { return fetchQuery(repo, qualifier); });
            });
        });

        setCounts([]);
        renderSkeleton();
        setStatus("Loading GitHub results...", "loading");
        if (refreshBtn) {
            refreshBtn.disabled = true;
        }

        fetchSequentially(queries).then(function (responses) {
            var items = normalizeItems(responses);
            writeCachedItems(items);
            render(items);
        })["catch"](function (error) {
            var cached = readCachedItems();
            if (cached) {
                render(cached.items);
                setStatus("Showing cached GitHub results from " + formatDate(cached.savedAt) + ". Live refresh failed: " + error.message, "error");
                return;
            }
            setCounts([]);
            resultsEl.innerHTML = "";
            if (error.status === 403) {
                setStatus("GitHub API rate limit reached for this browser/network. Try again after the public API quota resets.", "error");
            } else {
                setStatus("GitHub results could not be loaded: " + error.message, "error");
            }
        })["finally"](function () {
            if (refreshBtn) {
                refreshBtn.disabled = false;
            }
        });
    }

    filterEls.forEach(function (button) {
        button.addEventListener("click", function () {
            activeFilter = button.getAttribute("data-filter") || "all";
            render(allItems);
        });
    });

    function toggleRepo(button) {
        var repo = button.getAttribute("data-repo-toggle");
        var panel = document.getElementById(button.getAttribute("aria-controls"));
        var repoEl = button.closest ? button.closest(".osw-repo") : null;
        var expanded = button.getAttribute("aria-expanded") !== "true";

        expandedRepos[repo] = expanded;
        button.setAttribute("aria-expanded", expanded ? "true" : "false");
        if (panel) {
            panel.hidden = !expanded;
        }
        if (repoEl) {
            repoEl.classList.toggle("is-open", expanded);
        }
    }

    resultsEl.addEventListener("click", function (event) {
        if (!event.target.closest) {
            return;
        }

        var rowLink = event.target.closest(".osw-row");
        if (rowLink) {
            return;
        }

        var shell = event.target.closest("[data-repo-shell]");
        if (!shell || !resultsEl.contains(shell)) {
            return;
        }

        var button = shell.querySelector("[data-repo-toggle]");
        if (button) {
            toggleRepo(button);
        }
    });

    if (refreshBtn) {
        refreshBtn.addEventListener("click", loadResults);
    }

    loadResults();
}());
