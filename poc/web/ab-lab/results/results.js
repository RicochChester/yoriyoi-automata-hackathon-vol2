(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    runMeta: $("runMeta"), errorPanel: $("errorPanel"), errorMessage: $("errorMessage"),
    content: $("analysisContent"), conditionATitle: $("conditionATitle"), conditionBTitle: $("conditionBTitle"),
    conditionASetup: $("conditionASetup"), conditionBSetup: $("conditionBSetup"), metrics: $("metricsBody"),
    analysisText: $("analysisText"), footerMeta: $("footerMeta"),
  };

  function queryParams() {
    try { return new URLSearchParams(window.location.search); } catch (_) { return null; }
  }

  function showError(message) {
    els.content.hidden = true;
    els.errorPanel.hidden = false;
    els.errorMessage.textContent = message;
    els.runMeta.textContent = "保存済み履歴を読み込めませんでした";
  }

  async function fetchHistory(historyId) {
    const response = await fetch(`/api/ab-poc/history/${encodeURIComponent(historyId)}`, { headers: { Accept: "application/json" } });
    let payload;
    try { payload = await response.json(); } catch (_) { throw new Error(`HTTP ${response.status}`); }
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  function asComparison(payload) {
    if (payload?.runs?.A || payload?.runs?.B) return payload;
    const condition = payload?.condition_id || "A";
    return { ...payload, runs: { [condition]: payload }, metrics: { [condition]: payload.metrics || {} } };
  }

  function participantNames(comparison) {
    const people = comparison?.runs?.A?.participants || comparison?.runs?.B?.participants || [];
    return Object.fromEntries(people.map((person) => [person.id, person.name]));
  }

  function personNames(value, lookup) {
    if (value === undefined) return "—（保存値なし）";
    if (!Array.isArray(value) || !value.length) return "なし";
    return value.map((id) => lookup[id] || id).join("、");
  }

  function pathText(value, lookup) {
    if (value === undefined) return "—（保存値なし）";
    if (!Array.isArray(value) || !value.length) return "なし";
    return value.map((path) => `${lookup[path.from_known_endpoint] || path.from_known_endpoint}→${lookup[path.third_party] || path.third_party}（${path.stage || "—"}）`).join("、");
  }

  function knownPairText(value, lookup) {
    if (value === undefined) return "—（保存値なし）";
    if (value === null || typeof value !== "object") return "なし";
    const pair = Array.isArray(value.pair) ? value.pair.map((id) => lookup[id] || id).join("・") : "既知ペア";
    return `${pair}：${value.stage || "—"}`;
  }

  function rawNumber(value) { return typeof value === "number" && Number.isFinite(value) ? value : null; }
  function numericDiff(a, b, formatter = (value) => String(value)) {
    const left = rawNumber(a); const right = rawNumber(b);
    return left == null || right == null ? "—" : formatter(right - left);
  }
  function rateText(value) {
    const rate = rawNumber(value);
    return rate == null ? "—" : `${(rate * 100).toFixed(1)}%`;
  }
  function rateDiffText(a, b) {
    const left = rawNumber(a); const right = rawNumber(b);
    if (left == null || right == null) return "—";
    const sign = right - left >= 0 ? "+" : "";
    return `${sign}${((right - left) * 100).toFixed(1)}ポイント`;
  }

  function metricRows(comparison, lookup) {
    const a = comparison.metrics?.A || comparison.runs?.A?.metrics || {};
    const b = comparison.metrics?.B || comparison.runs?.B?.metrics || {};
    const turnsA = Array.isArray(comparison.runs?.A?.turns) ? comparison.runs.A.turns.length : null;
    const turnsB = Array.isArray(comparison.runs?.B?.turns) ? comparison.runs.B.turns.length : null;
    return [
      ["新しい顔見知り", a.new_acquaintance_count, b.new_acquaintance_count, numericDiff(a.new_acquaintance_count, b.new_acquaintance_count)],
      ["顔見知りの対象数", a.new_acquaintance_eligible_count, b.new_acquaintance_eligible_count, numericDiff(a.new_acquaintance_eligible_count, b.new_acquaintance_eligible_count)],
      ["顔見知り率", rateText(a.new_acquaintance_rate), rateText(b.new_acquaintance_rate), rateDiffText(a.new_acquaintance_rate, b.new_acquaintance_rate)],
      ["新しい親しみ", a.new_familiar_count, b.new_familiar_count, numericDiff(a.new_familiar_count, b.new_familiar_count)],
      ["孤立者（保存リスト）", personNames(a.isolated_participants, lookup), personNames(b.isolated_participants, lookup), "—"],
      ["既知ペアの対面化", knownPairText(a.known_pair_onsite, lookup), knownPairText(b.known_pair_onsite, lookup), "—"],
      ["第三者への観察経路", pathText(a.third_party_observation_paths, lookup), pathText(b.third_party_observation_paths, lookup), "—"],
      ["会話ターン数（保存ログ）", turnsA == null ? "—" : turnsA, turnsB == null ? "—" : turnsB, numericDiff(turnsA, turnsB)],
    ];
  }

  function render(comparison, envelope, historyId) {
    const runA = comparison.runs?.A;
    const runB = comparison.runs?.B;
    if (!runA || !runB) throw new Error("この履歴には条件A/Bの比較結果がありません。");
    const lookup = participantNames(comparison);
    els.conditionATitle.textContent = runA.condition?.label || "条件A";
    els.conditionBTitle.textContent = runB.condition?.label || "条件B";
    els.conditionASetup.textContent = runA.condition?.online_experience?.memory_summary || "保存された条件A";
    els.conditionBSetup.textContent = runB.condition?.online_experience?.memory_summary || "保存された条件B";
    els.runMeta.textContent = `seed ${comparison.root_seed ?? envelope.seed ?? "—"} / ${envelope.provider || "保存済み結果"} / 単一試行`;
    els.footerMeta.textContent = `履歴ID：${historyId}`;
    els.metrics.replaceChildren();
    metricRows(comparison, lookup).forEach(([label, a, b, diff]) => {
      const row = document.createElement("tr");
      [label, a, b, diff].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = String(value ?? "—");
        row.appendChild(cell);
      });
      els.metrics.appendChild(row);
    });
    const metricsA = comparison.metrics?.A || runA.metrics || {};
    const metricsB = comparison.metrics?.B || runB.metrics || {};
    const acquaintanceA = rawNumber(metricsA.new_acquaintance_count);
    const acquaintanceB = rawNumber(metricsB.new_acquaintance_count);
    const acquaintanceEligibleA = rawNumber(metricsA.new_acquaintance_eligible_count);
    const acquaintanceEligibleB = rawNumber(metricsB.new_acquaintance_eligible_count);
    const acquaintanceRateA = rawNumber(metricsA.new_acquaintance_rate);
    const acquaintanceRateB = rawNumber(metricsB.new_acquaintance_rate);
    const familiarA = rawNumber(metricsA.new_familiar_count);
    const familiarB = rawNumber(metricsB.new_familiar_count);
    const fragments = [];
    if ([acquaintanceA, acquaintanceB, acquaintanceEligibleA, acquaintanceEligibleB, acquaintanceRateA, acquaintanceRateB].every((value) => value != null)) {
      fragments.push(acquaintanceA === acquaintanceB && acquaintanceEligibleA !== acquaintanceEligibleB
        ? `顔見知り件数は同じ${acquaintanceA}件だが、対象数がA=${acquaintanceEligibleA}、B=${acquaintanceEligibleB}のため率は${rateText(acquaintanceRateA)}→${rateText(acquaintanceRateB)}`
        : `顔見知り件数はA=${acquaintanceA}件、B=${acquaintanceB}件、率は${rateText(acquaintanceRateA)}→${rateText(acquaintanceRateB)}`);
    } else {
      fragments.push("新しい顔見知りの保存値は比較できません");
    }
    fragments.push(familiarA != null && familiarB != null ? `新しい親しみはAが${familiarA}件、Bが${familiarB}件` : "新しい親しみの保存値は比較できません");
    if (Array.isArray(metricsB.third_party_observation_paths)) {
      fragments.push(metricsB.third_party_observation_paths.length ? "Bには保存された第三者への観察経路があります" : "Bに保存された第三者への観察経路はありません");
    }
    els.analysisText.textContent = `${fragments.join("。" )}。これは保存済み指標の差を短く記述したものです。`;
    els.errorPanel.hidden = true;
    els.content.hidden = false;
  }

  async function init() {
    const query = queryParams();
    const historyId = query?.get("history")?.trim() || "";
    if (!historyId) { showError("URLに history=<履歴ID> を指定してください。"); return; }
    try {
      const envelope = await fetchHistory(historyId);
      render(asComparison(envelope.result), envelope, historyId);
    } catch (error) {
      showError(`保存済み履歴を取得できませんでした：${error.message}`);
    }
  }

  init();
})();
