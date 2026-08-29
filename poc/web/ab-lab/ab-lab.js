(() => {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const positions = {
    akane: [200, 150],
    midori: [800, 150],
    koharu: [200, 425],
    kurumi: [800, 425],
  };
  const state = {
    comparison: null, range: "single", loadedRange: null, condition: "A", provider: "deepseek_autonomous",
    loadedProvider: null, step: 0, timer: null, sample: false, history: [],
    running: false, runStartedAt: 0, runElapsedTimer: null, resultTimer: null,
    resultPreviousFocus: null, historyId: "", demo: false, demoHistoryId: "", demoError: false,
  };
  const $ = (id) => document.getElementById(id);
  function parseDemoQuery() {
    try {
      const params = new URLSearchParams(window.location.search);
      return {
        enabled: params.get("demo") === "1",
        historyId: (params.get("history") || "").trim(),
      };
    } catch (_) {
      return { enabled: false, historyId: "" };
    }
  }
  const demoQuery = parseDemoQuery();
  state.demo = demoQuery.enabled;
  state.demoHistoryId = demoQuery.historyId;
  document.body.classList.toggle("demo-mode", state.demo);
  const providerLabel = (mode) => ({
    rule: "ルール（テスト用・再現可能）",
    ollama: "LLM（旧ローカル・API互換）",
    ollama_split: "LLM（ローカル・一括生成）",
    ollama_two_stage: "LLM（ローカル・二段生成）",
    ollama_planner_deepseek: "LLM（ローカル計画＋発話）",
    rule_candidates_deepseek: "LLM（クラウド・候補生成）",
    deepseek_natural: "LLM（自然会話・試験）",
    deepseek_autonomous: "LLM（自律会話・ナラティブ）",
  }[mode] || "—");
  const els = {
    seedForm: $("seedForm"), seed: $("seedInput"), provider: $("providerSelect"), range: $("rangeSelect"),
    run: $("runButton"), status: $("status"), progress: $("runProgress"), controlPanel: document.querySelector(".control-panel"),
    history: $("historySelect"), loadHistory: $("loadHistoryButton"),
    turnLabel: $("turnLabel"), eventNumber: $("eventNumber"), event: $("eventCard"),
    layer: $("relationLayer"), prev: $("prevButton"), play: $("playButton"), next: $("nextButton"),
    log: $("turnLog"), metrics: $("metricsBody"), seedLabel: $("seedLabel"), resultButton: $("resultButton"),
    analysisLink: $("analysisLink"),
    participantProvider: $("participantProvider"), participantModel: $("participantModel"), promptVersion: $("promptVersion"),
    participantAudit: $("participantAudit"), participantLatency: $("participantLatency"), fallbackWarning: $("fallbackWarning"),
    talkInitiator: $("talkInitiator"), responseProvider: $("responseProvider"),
    cautionText: $("cautionText"), relationshipEvaluator: $("relationshipEvaluator"),
    configVersion: $("configVersion"), runShape: $("runShape"), relationshipThresholds: $("relationshipThresholds"),
    rootSeed: $("rootSeed"), sampleBadge: $("sampleBadge"), footerMeta: $("footerMeta"),
    adoptedCount: $("adoptedCount"), hybridCount: $("hybridCount"),
    ruleFallbackCount: $("ruleFallbackCount"), fallbackStages: $("fallbackStages"),
    metricAHeader: $("metricAHeader"), metricBHeader: $("metricBHeader"),
    conversationModeRow: $("conversationModeRow"), conversationMode: $("conversationMode"),
    affectModeRow: $("affectModeRow"), affectMode: $("affectMode"),
    comparisonSummaryRow: $("comparisonSummaryRow"), comparisonSummary: $("comparisonSummary"),
    summaryParticipants: $("summaryParticipants"), summaryTurns: $("summaryTurns"),
    summaryUtterances: $("summaryUtterances"), summaryProvider: $("summaryProvider"),
    narrativeContent: $("narrativeContent"), narrativeCondition: $("narrativeCondition"),
    demoLabel: $("demoLabel"), demoError: $("demoError"), demoControls: $("demoControls"),
    resultOverlay: $("resultOverlay"), resultDialog: $("resultDialog"), resultCloseButton: $("resultCloseButton"),
    resultContinueButton: $("resultContinueButton"), resultReplayButton: $("resultReplayButton"),
    resultNote: $("resultNote"), resultCondition: $("resultCondition"), resultProvider: $("resultProvider"),
    resultTurns: $("resultTurns"), resultUtterances: $("resultUtterances"), resultAcquaintance: $("resultAcquaintance"),
    resultFamiliar: $("resultFamiliar"), resultIsolated: $("resultIsolated"), resultNarrativeTitle: $("resultNarrativeTitle"),
    resultNarrativeSummary: $("resultNarrativeSummary"), resultNarrativeEnding: $("resultNarrativeEnding"),
  };

  function setStatus(message, error = false) {
    els.status.textContent = message;
    els.status.className = error ? "status error" : "status";
    els.status.setAttribute("aria-busy", String(state.running));
  }

  function runningStatus(seed, elapsedSeconds) {
    const scope = state.range === "single"
      ? "単条件を生成しています。"
      : "A/B比較を生成しています。A・Bの結果は完了後に一括表示します。";
    return `seed ${seed} を${providerLabel(state.provider)}で実行中…（経過 ${elapsedSeconds}秒）${scope}実データの進行率は取得できません。`;
  }

  function updateRunningStatus(seed) {
    if (!state.running) return;
    const elapsedSeconds = Math.max(0, Math.floor((Date.now() - state.runStartedAt) / 1000));
    const message = runningStatus(seed, elapsedSeconds);
    setStatus(message);
    els.progress.setAttribute("aria-valuetext", message);
  }

  function restoreConditionControls() {
    document.querySelectorAll("[data-condition]").forEach((button) => {
      const disabled = state.running
        || ((state.loadedRange || state.range) === "single" && button.dataset.condition === "B");
      button.disabled = disabled;
      button.setAttribute("aria-disabled", String(disabled));
    });
  }

  function setRunControlsDisabled(disabled) {
    [els.provider, els.range, els.seed, els.run].forEach((control) => {
      if (control) control.disabled = disabled;
    });
    els.history.disabled = disabled || state.history.length === 0;
    els.loadHistory.disabled = disabled || !els.history.value;
    [els.prev, els.play, els.next].forEach((control) => {
      if (control) control.disabled = disabled || control.disabled;
    });
    restoreConditionControls();
  }

  function setRunning(running, seed) {
    if (running) {
      const runSeed = seed ?? els.seed.value;
      state.running = true;
      state.runStartedAt = Date.now();
      els.controlPanel?.setAttribute("aria-busy", "true");
      els.progress.hidden = false;
      els.progress.setAttribute("aria-busy", "true");
      setRunControlsDisabled(true);
      updateRunningStatus(runSeed);
      state.runElapsedTimer = setInterval(() => updateRunningStatus(runSeed), 1000);
      return;
    }
    if (state.runElapsedTimer) {
      clearInterval(state.runElapsedTimer);
      state.runElapsedTimer = null;
    }
    state.running = false;
    state.runStartedAt = 0;
    els.status.setAttribute("aria-busy", "false");
    els.controlPanel?.setAttribute("aria-busy", "false");
    els.progress.hidden = true;
    els.progress.setAttribute("aria-busy", "false");
    els.progress.setAttribute("aria-valuetext", "");
    setRunControlsDisabled(false);
    render();
  }

  async function fetchJson(url) {
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    let payload;
    try { payload = await response.json(); } catch (_) { throw new Error(`HTTP ${response.status}`); }
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  function apiRunUrl(path, seed, condition) {
    const params = new URLSearchParams({
      ...(condition ? { condition } : {}),
      seed: String(seed),
      participant_provider: state.provider,
      ...(state.provider !== "rule" ? { record_history: "1" } : {}),
    });
    // Selecting the provider alone never runs it. This flag is attached only
    // after the user explicitly submits the form with the autonomous mode.
    if (state.provider === "deepseek_autonomous") {
      params.set("external_consent", "deepseek");
    }
    return `${path}?${params.toString()}`;
  }

  function asComparison(payload) {
    if (payload?.runs?.A || payload?.runs?.B) return payload;
    const condition = payload?.condition_id || "A";
    return {
      ...payload,
      runs: { [condition]: payload },
      metrics: { [condition]: payload.metrics || {} },
    };
  }

  function runTurnCount(run) {
    return Array.isArray(run?.turns) ? run.turns.length : null;
  }

  function resultRangeLabel(comparison, range) {
    const countA = runTurnCount(comparison?.runs?.A);
    if (range === "single") {
      return countA == null ? "条件Aのみ" : `条件Aのみ・${countA}ターン`;
    }
    const countB = runTurnCount(comparison?.runs?.B);
    if (countA == null || countB == null) return "A/B比較";
    return countA === countB
      ? `A/B各${countA}ターン`
      : `A${countA}ターン / B${countB}ターン`;
  }

  const activeRun = () => state.comparison?.runs?.[state.condition] || null;
  const names = () => Object.fromEntries((activeRun()?.participants || []).map((person) => [person.id, person.name]));
  const personName = (id) => names()[id] || id || "—";

  const actLabels = {
    light_question: "軽い質問",
    self_disclosure: "自分の考え・気持ち",
    invitation: "一緒にやる提案",
    follow_up: "前の話の続き",
  };
  const fallbackLabels = {
    planner: "会話の計画づくり",
    utterance: "発話づくり",
    approach_renderer: "安全表現への置換",
    selection_utterance: "全体ルール切替",
  };
  const conversationModeLabels = {
    rule_no_affect: "ルール・主観なし",
    llm_no_affect: "LLM・主観なし",
    llm_affect: "LLM・主観あり",
    deepseek_natural: "LLM・自然会話",
    deepseek_autonomous: "LLM・自律会話・ナラティブ",
  };
  const affectModeLabels = { enabled: "あり", disabled: "なし" };
  const reactionLabels = {
    positive: "前向き",
    neutral: "ふつう",
    misaligned: "噛み合わない",
  };
  const verbalMovementLabels = {
    "-1": "後退",
    "0": "変化なし",
    "1": "前進",
  };
  const verbalMomentumLabels = {
    "0": "なし",
    "1": "弱い",
    "2": "維持",
    "3": "高い",
  };
  const episodeEndReasonLabels = {
    responder_end: "相手が終了",
    initiator_end: "話者が終了",
    utterance_limit: "4発話上限",
    provider_unavailable: "発話生成を継続できず終了",
  };

  function auditForTurn(turn) {
    return (activeRun()?.participant_audit?.turns || []).find((item) => item.turn === turn) || null;
  }

  function snapshotFor(run, step) {
    if (!run) return null;
    if (step === 0) return run.snapshots.find((item) => item.phase === "start") || run.snapshots[0];
    return run.snapshots.find((item) => item.phase === "turn_end" && item.turn === step) || null;
  }

  function makeLine(left, right, className, offset = 0) {
    const start = positions[left];
    const end = positions[right];
    if (!start || !end) return;
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const length = Math.hypot(dx, dy) || 1;
    const nx = -dy / length;
    const ny = dx / length;
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", start[0] + nx * offset); line.setAttribute("y1", start[1] + ny * offset);
    line.setAttribute("x2", end[0] + nx * offset); line.setAttribute("y2", end[1] + ny * offset);
    line.setAttribute("class", `edge ${className}`);
    els.layer.appendChild(line);
  }

  function renderGraph(run, turn) {
    els.layer.replaceChildren();
    document.querySelectorAll("[data-person]").forEach((node) => node.classList.remove("actor", "target"));
    const snapshot = snapshotFor(run, state.step);
    (snapshot?.pairs || []).forEach((pair) => {
      const hasOnline = pair.online_known === "あり";
      const hasOnsite = ["顔見知り", "親しみがある"].includes(pair.onsite_relationship);
      if (hasOnline) makeLine(pair.left_id, pair.right_id, "online", hasOnsite ? -6 : 0);
      if (pair.onsite_relationship === "顔見知り") makeLine(pair.left_id, pair.right_id, "acquaintance", hasOnline ? 6 : 0);
      if (pair.onsite_relationship === "親しみがある") makeLine(pair.left_id, pair.right_id, "familiar", hasOnline ? 6 : 0);
    });
    if (turn) {
      makeLine(turn.actor, turn.target, "talking");
      const personNodes = Array.from(document.querySelectorAll("[data-person]"));
      personNodes.find((node) => node.dataset.person === turn.actor)?.classList.add("actor");
      personNodes.find((node) => node.dataset.person === turn.target)?.classList.add("target");
    }
  }

  function appendText(parent, className, label, value) {
    const row = document.createElement("p");
    row.className = className;
    row.textContent = label ? `${label}${value}` : value;
    parent.appendChild(row);
  }

  function renderAffectTransition(turn, parent = els.event) {
    if (!turn || !Object.prototype.hasOwnProperty.call(turn, "affect_transition")) return;
    const transition = turn.affect_transition;
    if (!transition || !transition.direction) {
      appendText(parent, "affect-transition muted", "主観：", "主観更新なし");
      return;
    }
    const direction = transition.direction;
    const receiver = personName(direction.receiver);
    const initiator = personName(direction.initiator);
    const before = transition.before ?? "—";
    const after = transition.after ?? "—";
    const reaction = reactionLabels[transition.reaction] || transition.reaction || turn.reaction || "—";
    const fallback = turn.fallback || {};
    const initiatorFallback = Boolean(fallback.initiator || auditForTurn(turn.turn)?.fallback?.initiator);
    const responderFallback = Boolean(
      fallback.response || fallback.responder || auditForTurn(turn.turn)?.fallback?.responder,
    );
    const fallbackRoles = [
      initiatorFallback ? "話しかけ役" : "",
      responderFallback ? "応答役" : "",
    ].filter(Boolean);
    const fallbackText = fallbackRoles.length
      ? ` / fallback：${fallbackRoles.join("・")}`
      : "";
    appendText(
      parent,
      "affect-transition",
      "主観：",
      `${receiver} → ${initiator}  ${before} → ${after}（反応：${reaction}${fallbackText}）`,
    );
  }

  function episodeUtterances(turn) {
    if (!turn || turn.episode == null) return [];
    const source = Array.isArray(turn.episode)
      ? turn.episode
      : Array.isArray(turn.episode.utterances) ? turn.episode.utterances : [];
    return source
      .map((utterance, index) => ({ utterance, index }))
      .sort((left, right) => {
        const leftSequence = Number(left.utterance?.sequence);
        const rightSequence = Number(right.utterance?.sequence);
        const leftValid = Number.isFinite(leftSequence);
        const rightValid = Number.isFinite(rightSequence);
        if (leftValid && rightValid && leftSequence !== rightSequence) return leftSequence - rightSequence;
        if (leftValid !== rightValid) return leftValid ? -1 : 1;
        return left.index - right.index;
      })
      .map(({ utterance }) => utterance);
  }

  function hasEpisode(turn) {
    return Boolean(turn && turn.episode != null);
  }

  function episodeSequence(utterance, fallback) {
    const sequence = Number(utterance?.sequence);
    return Number.isInteger(sequence) && sequence > 0 ? sequence : fallback;
  }

  function episodeText(utterance) {
    return utterance?.text || "（発話本文なし）";
  }

  function episodeSummary(turn, max = 42) {
    const utterances = episodeUtterances(turn);
    if (!utterances.length) return "0発話";
    return `${utterances.length}発話  ${utterances.map((utterance, index) => (
      `#${episodeSequence(utterance, index + 1)} ${personName(utterance?.speaker)}：「${excerpt(episodeText(utterance), max)}」`
    )).join("  ")}`;
  }

  function runUtteranceCount(run) {
    return (Array.isArray(run?.turns) ? run.turns : []).reduce((total, turn) => (
      total + (hasEpisode(turn) ? episodeUtterances(turn).length : 2)
    ), 0);
  }

  function renderVerbalEvaluation(turn, parent) {
    const evaluation = turn?.verbal_evaluation;
    if (!evaluation || typeof evaluation !== "object") return;
    const movement = evaluation.relationship_movement
      ?? evaluation.relationshipMovement
      ?? evaluation.relationship_change
      ?? evaluation.movement;
    const momentum = evaluation.end_momentum
      ?? evaluation.endMomentum
      ?? evaluation.momentum;
    const movementNumber = Number(movement);
    const momentumNumber = Number(momentum);
    const movementText = Number.isInteger(movementNumber) && movementNumber >= -1 && movementNumber <= 1
      ? `${movementNumber > 0 ? "+" : ""}${movementNumber}（${verbalMovementLabels[String(movementNumber)]}）`
      : null;
    const momentumText = Number.isInteger(momentumNumber) && momentumNumber >= 0 && momentumNumber <= 3
      ? `${momentumNumber}（${verbalMomentumLabels[String(momentumNumber)]}）`
      : null;
    const reason = evaluation.reason || evaluation.concise_reason || evaluation.evaluation_reason;
    if (movementText) appendText(parent, "verbal-evaluation", "関係の動き：", movementText);
    if (momentumText) appendText(parent, "verbal-evaluation", "終了時の熱量：", momentumText);
    if (reason) appendText(parent, "verbal-evaluation-reason", "理由：", String(reason));
  }

  function renderEpisodeEndReason(turn, parent) {
    if (!hasEpisode(turn)) return;
    const label = episodeEndReasonLabels[turn.episode_end_reason];
    if (label) appendText(parent, "episode-end-reason", "終了：", label);
  }

  function renderEvent(turn) {
    els.event.replaceChildren();
    if (!turn) {
      const idle = document.createElement("p");
      idle.className = "idle-message";
      idle.textContent = "まだ会話は始まっていません。開始時の関係を表示中です。";
      els.event.appendChild(idle);
      els.eventNumber.textContent = "T00";
      return;
    }
    els.eventNumber.textContent = `T${String(turn.turn).padStart(2, "0")}`;
    const audit = auditForTurn(turn.turn);
    appendText(els.event, "event-pair", "", `${personName(turn.actor)} → ${personName(turn.target)}`);
    appendText(els.event, "event-topic", "話題：", turn.topic);
    const speech = document.createElement("div");
    speech.className = "speech-stack";
    if (hasEpisode(turn)) {
      episodeUtterances(turn).forEach((utterance, index) => {
        const bubble = document.createElement("div");
        const speaker = utterance?.speaker;
        bubble.className = `speech-bubble episode-speech ${speaker === turn.actor ? "actor-speech" : speaker === turn.target ? "target-speech" : ""}`;
        const label = document.createElement("strong");
        label.textContent = `${personName(speaker)} #${episodeSequence(utterance, index + 1)}`;
        const text = document.createElement("p");
        text.textContent = episodeText(utterance);
        bubble.append(label, text);
        speech.appendChild(bubble);
      });
    } else {
      const actorBubble = document.createElement("div");
      actorBubble.className = "speech-bubble actor-speech";
      const actorLabel = document.createElement("strong");
      actorLabel.textContent = `${personName(turn.actor)}（働きかけ）`;
      const actorText = document.createElement("p");
      actorText.textContent = turn.approach || "（働きかけの本文なし）";
      actorBubble.append(actorLabel, actorText);
      const targetBubble = document.createElement("div");
      targetBubble.className = "speech-bubble target-speech";
      const targetLabel = document.createElement("strong");
      targetLabel.textContent = `${personName(turn.target)}（返答）`;
      const targetText = document.createElement("p");
      targetText.textContent = turn.reply || "（返答本文なし）";
      targetBubble.append(targetLabel, targetText);
      speech.append(actorBubble, targetBubble);
    }
    els.event.appendChild(speech);

    const details = document.createElement("details");
    details.className = "event-details";
    const detailsSummary = document.createElement("summary");
    detailsSummary.textContent = "反応・主観・評価（補足）";
    details.appendChild(detailsSummary);
    if (audit?.adopted_plan) {
      const plan = audit.adopted_plan;
      const act = actLabels[plan.act] || plan.act || "—";
      appendText(details, "event-plan", "LLM採用：", `${personName(plan.target_id)} / ${plan.topic} / ${act}`);
      if (audit.outcome === "hybrid") {
        appendText(details, "event-plan fallback", "安全表現への置換：", "LLMの判断を維持し、発話を安全な定型表現へ置換");
      }
    } else if (audit?.outcome === "fallback") {
      const stage = fallbackLabels[audit.fallback_stage] || audit.fallback_stage || "段階情報なし";
      appendText(details, "event-plan fallback", "LLM計画：", `採用なし（${stage}へ切替）`);
    }
    const reaction = reactionLabels[turn.reaction] || turn.reaction || turn.response || "—";
    appendText(details, "response", "反応：", reaction);
    renderAffectTransition(turn, details);
    appendText(details, "evaluation", "第三者評価：", turn.evaluation?.reason || turn.evaluation_reason || "—");
    renderVerbalEvaluation(turn, details);
    renderEpisodeEndReason(turn, details);
    els.event.appendChild(details);
  }

  function excerpt(value, max = 42) {
    const text = String(value || "").replace(/\s+/g, " ").trim();
    if (!text) return "本文なし";
    return text.length > max ? `${text.slice(0, max - 1)}…` : text;
  }

  function renderLog(run) {
    els.log.replaceChildren();
    run.turns.forEach((turn) => {
      const item = document.createElement("li");
      if (turn.turn === state.step) item.className = "selected";
      const button = document.createElement("button");
      button.type = "button";
      if (turn.turn === state.step) button.setAttribute("aria-current", "step");
      const number = document.createElement("span");
      number.className = "turn-no";
      number.textContent = `T${String(turn.turn).padStart(2, "0")}`;
      const summary = document.createElement("span");
      summary.className = "turn-summary";
      summary.textContent = hasEpisode(turn)
        ? `${personName(turn.actor)} → ${personName(turn.target)}  ${episodeSummary(turn)}`
        : `${personName(turn.actor)} → ${personName(turn.target)}  「${excerpt(turn.approach)}」  ↩ 「${excerpt(turn.reply)}」`;
      button.append(number, summary);
      button.addEventListener("click", () => { stop(); state.step = turn.turn; render(); });
      item.appendChild(button);
      els.log.appendChild(item);
    });
    els.log.querySelector(".selected")?.scrollIntoView({ block: "nearest" });
  }

  function metricValue(metrics, key) {
    if (!metrics) return "—";
    if (key === "acquaintance") return `${metrics.new_acquaintance_count} / ${metrics.new_acquaintance_eligible_count}（割合 ${metrics.new_acquaintance_rate ?? "—"}）`;
    if (key === "familiar") return String(metrics.new_familiar_count);
    if (key === "known") return metrics.known_pair_onsite?.stage || "N/A";
    if (key === "isolated") return (metrics.isolated_participants || []).map(personName).join("、") || "なし";
    if (key === "paths") {
      const paths = metrics.third_party_observation_paths;
      if (paths == null) return "N/A";
      if (!Array.isArray(paths) || paths.length === 0) return "なし";
      return paths.map((path) => `${personName(path.from_known_endpoint)}→${personName(path.third_party)}（${path.stage ?? "—"}）`).join("、");
    }
    return "—";
  }

  function renderMetrics() {
    const metrics = state.comparison?.metrics || {};
    const single = (state.loadedRange || state.range) === "single";
    els.metricAHeader.textContent = single ? "条件A" : "A：直接既知なし";
    els.metricBHeader.textContent = single ? "比較なし" : "B：既知1組";
    const rows = [
      ["新規顔見知り", "acquaintance"],
      ["新規の親しみ", "familiar"],
      ["既知ペアの対面化", "known"],
      ["関係上の孤立者", "isolated"],
      ["第三者への観察経路", "paths"],
    ];
    els.metrics.replaceChildren();
    rows.forEach(([label, key]) => {
      const tr = document.createElement("tr");
      const bValue = single ? "—（比較なし）" : metricValue(metrics.B, key);
      [label, metricValue(metrics.A, key), bValue].forEach((value) => {
        const cell = document.createElement(label === value ? "th" : "td");
        cell.textContent = value;
        tr.appendChild(cell);
      });
      els.metrics.appendChild(tr);
    });
    els.seedLabel.textContent = single
      ? `条件Aのみ・seed ${state.comparison?.root_seed ?? "—"}`
      : `このseedのA/B比較（${state.comparison?.root_seed ?? "—"}）`;
  }

  function renderRunSummary(run) {
    const participants = run?.participants?.length ?? 0;
    const turns = run?.turns?.length ?? 0;
    els.summaryParticipants.textContent = `${participants || "—"}人`;
    els.summaryTurns.textContent = `${turns || "—"}ターン`;
    const utterances = runUtteranceCount(run);
    els.summaryUtterances.textContent = turns ? `${utterances}発話` : "—";
    els.summaryProvider.textContent = providerLabel(state.loadedProvider || state.provider);
  }

  function clearResultTimer() {
    if (state.resultTimer) {
      clearTimeout(state.resultTimer);
      state.resultTimer = null;
    }
  }

  function updateDemoLabel(run = activeRun()) {
    if (!state.demo || !els.demoLabel || !run) return;
    els.demoLabel.hidden = false;
    els.demoLabel.textContent = `DEMO / CONDITION ${state.condition} / TURN ${state.step}/${run.turns.length}`;
  }

  function updateAnalysisLink() {
    if (!els.analysisLink) return;
    const hasHistory = Boolean(
      state.historyId
      && state.loadedRange === "comparison"
      && state.comparison?.runs?.A
      && state.comparison?.runs?.B
    );
    els.analysisLink.hidden = !hasHistory;
    if (hasHistory) {
      els.analysisLink.href = `results/?history=${encodeURIComponent(state.historyId)}`;
    }
  }

  function setDemoError(message) {
    clearResultTimer();
    stop();
    state.demoError = true;
    if (els.demoError) {
      els.demoError.hidden = false;
      els.demoError.textContent = `DEMOエラー：${message}`;
    }
    setStatus(`DEMOエラー：${message}`, true);
  }

  function closeResult({ restoreFocus = true } = {}) {
    clearResultTimer();
    if (!els.resultOverlay) return;
    const previousFocus = state.resultPreviousFocus;
    state.resultPreviousFocus = null;
    els.resultOverlay.hidden = true;
    if (restoreFocus && previousFocus && typeof previousFocus.focus === "function" && document.contains(previousFocus)) {
      previousFocus.focus();
    }
  }

  function resultMetrics(run) {
    return run?.metrics || state.comparison?.metrics?.[state.condition] || {};
  }

  function renderResultContent() {
    const comparison = state.comparison;
    const run = activeRun();
    if (!comparison || !run) return false;
    const metrics = resultMetrics(run);
    const condition = run.condition?.label || comparison.conditions?.[state.condition]?.label || `条件${state.condition}`;
    const narrative = narrativeFor(comparison, run);
    const turns = Array.isArray(run.turns) ? run.turns.length : 0;
    els.resultCondition.textContent = condition;
    els.resultProvider.textContent = providerLabel(state.loadedProvider || state.provider);
    els.resultTurns.textContent = `${turns}ターン`;
    els.resultUtterances.textContent = `${runUtteranceCount(run)}発話`;
    els.resultAcquaintance.textContent = `${metrics.new_acquaintance_count ?? "—"} / ${metrics.new_acquaintance_eligible_count ?? "—"}`;
    els.resultFamiliar.textContent = String(metrics.new_familiar_count ?? "—");
    els.resultIsolated.textContent = Array.isArray(metrics.isolated_participants) && metrics.isolated_participants.length
      ? metrics.isolated_participants.map(personName).join("、")
      : "なし";
    els.resultNarrativeTitle.textContent = narrative.title || "会話ログの記録";
    els.resultNarrativeSummary.textContent = narrative.summary || "会話ログから今回のやり取りをまとめています。";
    els.resultNarrativeEnding.textContent = narrative.ending || "会話ログの最後まで表示しました。";
    els.resultNote.textContent = "このログから読み取れる範囲のまとめです。人の行動を予測・保証するものではありません。";
    return true;
  }

  function openResult() {
    const run = activeRun();
    if (!run || state.step < run.turns.length || !renderResultContent()) return;
    clearResultTimer();
    state.resultPreviousFocus = document.activeElement;
    els.resultOverlay.hidden = false;
    els.resultDialog?.focus();
  }

  function scheduleAutoResult() {
    if (state.demo) return;
    const run = activeRun();
    if (!run || state.step < run.turns.length) return;
    clearResultTimer();
    state.resultTimer = setTimeout(() => {
      state.resultTimer = null;
      if (!state.timer && activeRun() === run && state.step >= run.turns.length) {
        openResult();
      }
    }, 2500);
  }

  function updateResultButton(run) {
    if (!els.resultButton) return;
    const atEnd = Boolean(run && state.step >= run.turns.length);
    els.resultButton.hidden = !atEnd;
    els.resultButton.disabled = !atEnd;
  }

  function fallbackNarrative(run) {
    const turns = Array.isArray(run?.turns) ? run.turns : [];
    const moments = turns.slice(0, 4).map((turn) => (
      hasEpisode(turn)
        ? `T${String(turn.turn).padStart(2, "0")} ${personName(turn.actor)} → ${personName(turn.target)}：${episodeSummary(turn, 52)}`
        : `T${String(turn.turn).padStart(2, "0")} ${personName(turn.actor)}：「${excerpt(turn.approach, 52)}」→ ${personName(turn.target)}：「${excerpt(turn.reply, 52)}」`
    ));
    const utterances = runUtteranceCount(run);
    const last = turns[turns.length - 1];
    return {
      title: "会話ログの記録",
      summary: `${turns.length}ターン、${utterances}発話の実際のやり取りを記録しています。`,
      moments,
      ending: last
        ? hasEpisode(last)
          ? `T${String(last.turn).padStart(2, "0")}の${episodeUtterances(last).length}発話で、この会話ログは終了しました。`
          : `T${String(last.turn).padStart(2, "0")}の返答で、この会話ログは終了しました。`
        : "会話ログはまだありません。",
      fallback: true,
    };
  }

  function narrativeFor(comparison, run) {
    const scoped = comparison?.narrative?.[state.condition];
    const scopedMeta = comparison?.narrative_meta?.[state.condition];
    const direct = run?.narrative;
    const candidate = direct && typeof direct === "object" ? direct : (scoped && typeof scoped === "object" ? scoped : null);
    if (!candidate || Array.isArray(candidate)) return fallbackNarrative(run);
    const metadata = run?.narrative_meta || scopedMeta || {};
    return {
      title: String(candidate.title || "今回の出会い"),
      summary: String(candidate.summary || ""),
      moments: Array.isArray(candidate.moments) ? candidate.moments.map((moment) => String(moment)).filter(Boolean) : [],
      ending: String(candidate.ending || ""),
      source: typeof metadata.source === "string" ? metadata.source : "unknown",
      fallback: candidate.fallback === true,
    };
  }

  function renderNarrative() {
    const comparison = state.comparison;
    const run = activeRun();
    if (!comparison || !run || !els.narrativeContent) return;
    const narrative = narrativeFor(comparison, run);
    els.narrativeCondition.textContent = `条件${state.condition}`;
    els.narrativeContent.replaceChildren();
    const source = document.createElement("p");
    source.className = "narrative-source";
    source.textContent = narrative.source === "deepseek"
      ? "LLMによる物語"
      : narrative.source === "local_fallback"
        ? "ローカル要約（LLMから切替）"
        : narrative.source === "local"
          ? "ローカル要約"
          : "会話ログからの控えめな記録";
    const title = document.createElement("h3");
    title.textContent = narrative.title;
    els.narrativeContent.append(source, title);
    if (narrative.summary) {
      const summary = document.createElement("p");
      summary.className = "narrative-summary";
      summary.textContent = narrative.summary;
      els.narrativeContent.appendChild(summary);
    }
    if (narrative.moments.length) {
      const momentsLabel = document.createElement("h4");
      momentsLabel.textContent = "会話の瞬間";
      const moments = document.createElement("ul");
      moments.className = "narrative-moments";
      narrative.moments.forEach((moment) => {
        const item = document.createElement("li");
        item.textContent = moment;
        moments.appendChild(item);
      });
      els.narrativeContent.append(momentsLabel, moments);
    }
    if (narrative.ending) {
      const ending = document.createElement("p");
      ending.className = "narrative-ending";
      ending.textContent = narrative.ending;
      els.narrativeContent.appendChild(ending);
    }
  }

  function renderRunInfo() {
    const comparison = state.comparison;
    const run = activeRun();
    if (!comparison || !run) return;
    const providerValue = run.participant_provider || {};
    const provider = typeof providerValue === "string" ? { name: providerValue } : providerValue;
    const audit = run.participant_audit || comparison.participant_audit || {};
    const roles = audit.roles || {};
    const summary = audit.summary || {};
    const evaluator = run.relationship_evaluator || {};
    const relationship = run.rules?.relationship || {};
    const auditNumber = (...keys) => {
      for (const key of keys) {
        if (audit[key] !== undefined && audit[key] !== null) return audit[key];
      }
      return "—";
    };
    const model = auditNumber("model", "model_name") !== "—"
      ? auditNumber("model", "model_name") : (provider.model || "—");
    const prompt = auditNumber("prompt_version", "promptVersion");
    const successes = auditNumber("successes", "llm_successes", "successful_calls") !== "—"
      ? auditNumber("successes", "llm_successes", "successful_calls") : (summary.llm ?? "—");
    const attempts = summary.attempts ?? audit.attempts;
    const turns = summary.turns ?? run.turns?.length;
    const retries = auditNumber("retries", "llm_retries") !== "—"
      ? auditNumber("retries", "llm_retries")
      : (attempts !== undefined && turns !== undefined ? Math.max(0, attempts - turns) : "—");
    const fallbacks = auditNumber("fallbacks", "llm_fallbacks", "fallback_count") !== "—"
      ? auditNumber("fallbacks", "llm_fallbacks", "fallback_count") : (summary.fallback ?? "—");
    const measuredLatency = (audit.turns || []).flatMap((turnAudit) => turnAudit.attempts || [])
      .reduce((total, attempt) => total + (Number(attempt.elapsed_ms) || 0), 0);
    const latency = auditNumber("total_latency_ms", "latency_ms", "total_latency") !== "—"
      ? auditNumber("total_latency_ms", "latency_ms", "total_latency")
      : (measuredLatency || "—");
    const fallbackCount = Number(fallbacks);
    const auditTurns = Array.isArray(audit.turns) ? audit.turns : [];
    const adopted = auditTurns.filter((turnAudit) => turnAudit?.adopted_plan).length;
    const fallbackStageCounts = auditTurns.reduce((counts, turnAudit) => {
      if (["hybrid", "fallback"].includes(turnAudit?.outcome)) {
        const stage = turnAudit.fallback_stage || "unknown";
        counts[stage] = (counts[stage] || 0) + 1;
      }
      return counts;
    }, {});
    const fallbackStageText = Object.entries(fallbackStageCounts)
      .map(([stage, count]) => `${fallbackLabels[stage] || stage} ${count}回`)
      .join("、") || "なし";
    const loadedProvider = state.loadedProvider || state.provider;
    const loadedRange = state.loadedRange || state.range;
    const isRuleProvider = loadedProvider === "rule";
    const isDeepSeekProvider = ["rule_candidates_deepseek", "deepseek_natural", "deepseek_autonomous"].includes(loadedProvider);
    const isAutonomousProvider = loadedProvider === "deepseek_autonomous";
    const isTwoStageProvider = ["ollama_two_stage", "ollama_planner_deepseek"].includes(loadedProvider);
    const hasAdoptedPlanField = auditTurns.some((turnAudit) => Object.prototype.hasOwnProperty.call(turnAudit || {}, "adopted_plan"));
    const turnCount = run.turns?.length ?? 8;
    const fullLlmWording = Number(summary.full_llm_wording);
    const hybrid = Number(summary.hybrid);
    const wholeFallback = Number(summary.whole_fallback);
    const fallbackFlagCount = (role) => auditTurns.filter((turnAudit) => Boolean(turnAudit?.fallback?.[role])).length;
    const initiatorFallback = Number.isFinite(Number(summary.initiator_fallback))
      ? Number(summary.initiator_fallback) : fallbackFlagCount("initiator");
    const responderFallback = Number.isFinite(Number(summary.responder_fallback))
      ? Number(summary.responder_fallback) : fallbackFlagCount("responder");
    const autonomousFallbackTurns = auditTurns.filter((turnAudit) => (
      Boolean(turnAudit?.fallback?.initiator) || Boolean(turnAudit?.fallback?.responder)
    )).length;
    let adoptedText = "—";
    if (isRuleProvider) {
      adoptedText = "対象外";
    } else if (isAutonomousProvider) {
      adoptedText = `話しかけ ${Math.max(0, turnCount - initiatorFallback)}/${turnCount}・返答 ${Math.max(0, turnCount - responderFallback)}/${turnCount}`;
    } else if (isDeepSeekProvider) {
      adoptedText = `${(Number.isFinite(fullLlmWording) ? fullLlmWording : 0) + (Number.isFinite(hybrid) ? hybrid : 0)} / ${turnCount}`;
    } else if (isTwoStageProvider && hasAdoptedPlanField) {
      adoptedText = `${adopted} / ${turnCount}`;
    } else if (!isTwoStageProvider && Number.isFinite(Number(summary.llm))) {
      adoptedText = `${Number(summary.llm)} / ${turnCount}`;
    }
    let fallbackStageDisplay = fallbackStageText;
    if (isRuleProvider) {
      fallbackStageDisplay = "対象外";
    } else if (Number.isFinite(fallbackCount) && fallbackCount > 0 && Object.keys(fallbackStageCounts).length === 0) {
      fallbackStageDisplay = "段階情報なし";
    }
    const roleText = (role) => {
      const metadata = roles[role] || {};
      return [metadata.name, metadata.version].filter(Boolean).join(" / ") || "—";
    };
    els.participantProvider.textContent = [provider.name, provider.version].filter(Boolean).join(" / ") || "—";
    els.talkInitiator.textContent = roleText("initiator");
    els.responseProvider.textContent = roleText("responder");
    els.participantModel.textContent = String(model);
    els.promptVersion.textContent = String(prompt);
    els.participantAudit.textContent = isAutonomousProvider
      ? `自律会話 ${turnCount}ターン / 話しかけfallback ${initiatorFallback} / 返答fallback ${responderFallback}`
      : isDeepSeekProvider
      ? `LLM表現 ${Number.isFinite(fullLlmWording) ? fullLlmWording : 0} / 安全表現への置換 ${Number.isFinite(hybrid) ? hybrid : 0} / 全体ルール切替 ${Number.isFinite(wholeFallback) ? wholeFallback : 0}`
      : `成功 ${successes} / 再試行 ${retries} / フォールバック ${fallbacks}`;
    els.participantLatency.textContent = latency === "—" ? "—" : `${latency} ms`;
    els.adoptedCount.textContent = adoptedText;
    els.hybridCount.textContent = isRuleProvider || isAutonomousProvider
      ? "対象外"
      : isDeepSeekProvider ? `${Number.isFinite(hybrid) ? hybrid : 0} / ${turnCount}` : "—";
    els.ruleFallbackCount.textContent = isRuleProvider
      ? "対象外"
      : isAutonomousProvider ? `${autonomousFallbackTurns} / ${turnCount}`
      : `${isDeepSeekProvider && Number.isFinite(wholeFallback) ? wholeFallback : Number.isFinite(fallbackCount) ? fallbackCount : 0} / ${turnCount}`;
    els.fallbackStages.textContent = fallbackStageDisplay;
    const hasHybrid = !isAutonomousProvider && isDeepSeekProvider && Number.isFinite(hybrid) && hybrid > 0;
    const hasWholeFallback = isDeepSeekProvider
      ? isAutonomousProvider ? autonomousFallbackTurns > 0 : Number.isFinite(wholeFallback) && wholeFallback > 0
      : Number.isFinite(fallbackCount) && fallbackCount > 0;
    els.fallbackWarning.hidden = !(hasHybrid || hasWholeFallback);
    els.fallbackWarning.textContent = isAutonomousProvider
      ? `自律会話の一部でルールへ切替（${autonomousFallbackTurns}/${turnCount}ターン）。`
      : hasHybrid && hasWholeFallback
        ? "一部は安全表現へ置換され、一部は全体をルールへ切り替えました。"
        : hasHybrid ? "一部の発話を安全表現へ置換しました。" : "一部のターン全体をルールへ切り替えました。";
    els.relationshipEvaluator.textContent = [evaluator.name, evaluator.version].filter(Boolean).join(" / ") || "—";
    els.configVersion.textContent = run.config_version || comparison.config_version || "—";
    const selectedProviderLabel = providerLabel(loadedProvider);
    const rangeLabel = loadedRange === "single" ? "条件Aのみ" : "A/B各1試行";
    els.runShape.textContent = `${run.participants?.length ?? "—"}人 / ${run.turns?.length ?? "—"}ターン / ${rangeLabel} / ${selectedProviderLabel}`;
    els.relationshipThresholds.textContent = `顔見知り：会話${relationship.acquaintance_conversations ?? "—"}回・前向き${relationship.acquaintance_positive ?? "—"}回以上、親しみ：会話${relationship.familiar_conversations ?? "—"}回・前向き${relationship.familiar_positive ?? "—"}回以上`;
    els.rootSeed.textContent = String(comparison.root_seed ?? run.root_seed ?? "—");
    const modePayload = run;
    const hasConversationMode = typeof modePayload.conversation_mode === "string";
    const hasAffectMode = typeof modePayload.affect_mode === "string";
    els.conversationModeRow.hidden = !hasConversationMode;
    els.affectModeRow.hidden = !hasAffectMode;
    els.conversationMode.textContent = hasConversationMode
      ? (conversationModeLabels[modePayload.conversation_mode] || modePayload.conversation_mode)
      : "—";
    els.affectMode.textContent = hasAffectMode
      ? (affectModeLabels[modePayload.affect_mode] || modePayload.affect_mode)
      : "—";
    const comparisonDetails = run.comparison || comparison.comparison?.[state.condition];
    if (comparisonDetails && typeof comparisonDetails === "object") {
      const fallbacks = comparisonDetails.fallbacks || {};
      els.comparisonSummaryRow.hidden = false;
      els.comparisonSummary.textContent = [
        `既知pair会話 ${comparisonDetails.known_pair_conversations ?? "—"}`,
        `非既知pair ${comparisonDetails.non_direct_pair_count ?? "—"}`,
        `主観変化 ${comparisonDetails.affect_changes ?? "—"}`,
        `同方向の後続選択 ${(comparisonDetails.selections_after_affect_change || []).length}`,
        `fallback I:${fallbacks.initiator ?? 0} / R:${fallbacks.responder ?? 0}`,
      ].join(" ・ ");
    } else {
      els.comparisonSummaryRow.hidden = true;
      els.comparisonSummary.textContent = "—";
    }
    els.sampleBadge.hidden = !state.sample;
    els.footerMeta.textContent = `${run.participants?.length ?? "—"} PEOPLE / ${run.turns?.length ?? "—"} TURNS / ${rangeLabel} / ${selectedProviderLabel}`;
    renderRunSummary(run);
    renderNarrative();
    els.cautionText.replaceChildren();
    const cautionLabel = document.createElement("strong");
    cautionLabel.textContent = "読み方：";
    const caution = isRuleProvider
      ? `これは1つのseedによる${selectedProviderLabel}の再現可能な試行です。人の行動を予測するものではなく、因果を示すものではありません。`
      : `これは単一seedによる${selectedProviderLabel}の試行で、因果を示すものではありません。最小限の匿名情報をクラウドへ送信し、出力は非決定的です。APIキー未設定時は全ターンをルールで実行します。`;
    els.cautionText.append(cautionLabel, document.createTextNode(caution));
  }

  function render() {
    const run = activeRun();
    if (!run) return;
    const max = run.turns.length;
    state.step = Math.max(0, Math.min(state.step, max));
    const turn = state.step ? run.turns[state.step - 1] : null;
    els.turnLabel.textContent = state.step ? `TURN ${state.step} / ${max}` : `START / ${max}`;
    updateDemoLabel(run);
    renderGraph(run, turn);
    renderEvent(turn);
    renderLog(run);
    els.prev.disabled = state.step <= 0;
    els.next.disabled = state.step >= max;
    els.play.disabled = max === 0;
    els.play.textContent = state.timer ? "■ 停止" : "▶ 再生";
    updateResultButton(run);
    document.querySelectorAll("[data-condition]").forEach((button) => {
      const active = button.dataset.condition === state.condition;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      const disabled = state.running
        || ((state.loadedRange || state.range) === "single" && button.dataset.condition === "B");
      button.disabled = disabled;
      button.setAttribute("aria-disabled", String(disabled));
    });
  }

  function stop() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  function togglePlay() {
    if (!activeRun()) return;
    if (state.timer) { stop(); render(); return; }
    if (state.step >= activeRun().turns.length) state.step = 0;
    clearResultTimer();
    state.timer = setInterval(() => {
      if (state.step >= activeRun().turns.length) { stop(); render(); return; }
      state.step += 1;
      render();
      if (state.step >= activeRun().turns.length) {
        stop();
        render();
        scheduleAutoResult();
      }
    }, 2500);
    render();
  }

  function applyLoaded(comparison, { provider, range, sample = false, message = "" } = {}) {
    const requiredRun = range === "single" ? comparison.runs?.A : (comparison.runs?.A && comparison.runs?.B);
    if (!requiredRun) throw new Error(range === "single" ? "単条件結果の形式が正しくありません" : "A/B結果の形式が正しくありません");
    closeResult({ restoreFocus: false });
    state.comparison = comparison;
    state.loadedProvider = provider || state.provider;
    state.loadedRange = range || state.range;
    state.range = state.loadedRange;
    // Keep the public default on the generic LLM option when the bundled
    // fallback sample is shown. A saved run, however, selects its provider
    // so the loaded implementation is explicit in the controls.
    if (!sample && provider) state.provider = state.loadedProvider;
    state.sample = sample;
    state.demoError = false;
    if (state.demo || state.historyId) updateAnalysisLink();
    if (els.demoError) {
      els.demoError.hidden = true;
      els.demoError.textContent = "";
    }
    if (els.demoControls) els.demoControls.hidden = !state.demo;
    state.condition = "A";
    state.step = 0;
    els.provider.value = state.provider;
    els.range.value = state.range;
    renderMetrics();
    renderRunInfo();
    render();
    if (message) setStatus(message);
  }

  function renderHistoryOptions(entries) {
    els.history.replaceChildren();
    if (!entries.length) {
      const option = document.createElement("option");
      option.textContent = "保存済み履歴なし（再実行で保存）";
      option.value = "";
      els.history.appendChild(option);
      els.history.disabled = true;
      els.loadHistory.disabled = true;
      return;
    }
    entries.forEach((entry) => {
      const option = document.createElement("option");
      const date = entry.created_at ? new Date(entry.created_at).toLocaleString("ja-JP") : "日時不明";
      option.value = entry.id;
      option.textContent = `${date} / ${providerLabel(entry.provider)} / seed ${entry.seed ?? "—"}`;
      els.history.appendChild(option);
    });
    els.history.disabled = false;
    els.loadHistory.disabled = false;
  }

  async function refreshHistory() {
    try {
      const payload = await fetchJson("/api/ab-poc/history");
      state.history = Array.isArray(payload.history) ? payload.history : [];
      renderHistoryOptions(state.history);
      return state.history;
    } catch (_) {
      state.history = [];
      renderHistoryOptions([]);
      return [];
    }
  }

  async function loadHistoryEntry(historyId, { silent = false } = {}) {
    if (!historyId) return false;
    const envelope = await fetchJson(`/api/ab-poc/history/${encodeURIComponent(historyId)}`);
    const provider = envelope.provider || "deepseek_autonomous";
    const range = envelope.run_range === "single" ? "single" : "comparison";
    const comparison = asComparison(envelope.result);
    applyLoaded(comparison, {
      provider, range, sample: false,
      message: silent ? "" : `保存済み履歴を表示中：${providerLabel(provider)} / seed ${envelope.seed ?? comparison.root_seed}`,
    });
    state.historyId = String(historyId);
    els.seed.value = comparison.root_seed ?? envelope.seed ?? 1;
    updateAnalysisLink();
    return true;
  }

  async function loadInitial() {
    stop();
    state.historyId = "";
    updateAnalysisLink();
    if (state.demo) {
      setStatus("DEMO：指定された保存履歴を確認中…");
      if (!state.demoHistoryId) {
        setDemoError("URLに history=<履歴ID> を指定してください。自動再生は開始しません。");
        return;
      }
      const entries = await refreshHistory();
      const entry = entries.find((item) => String(item.id) === state.demoHistoryId);
      if (!entry) {
        setDemoError(`履歴「${state.demoHistoryId}」が一覧にありません。自動再生は開始しません。`);
        return;
      }
      try {
        await loadHistoryEntry(entry.id, { silent: true });
        setStatus("DEMO：保存済み履歴を読み込みました。手動操作で再生できます。");
      } catch (error) {
        setDemoError(`履歴「${state.demoHistoryId}」を読み込めませんでした：${error.message}`);
      }
      return;
    }
    setStatus("保存済みLLM履歴を確認中…");
    const entries = await refreshHistory();
    try {
      if (entries.length) {
        await loadHistoryEntry(entries[0].id, { silent: true });
        setStatus(`保存済みLLM履歴を表示中：${providerLabel(state.loadedProvider)} / seed ${state.comparison.root_seed}`);
        return;
      }
      const comparison = asComparison(await fetchJson("sample-run-ab-seed-1.json"));
      applyLoaded(comparison, {
        provider: "rule", range: "single", sample: true,
        message: "保存済みLLM履歴なし：APIを呼ばず固定サンプルを表示中（再実行で実行）",
      });
    } catch (error) {
      setStatus(`初期表示エラー：${error.message}`, true);
    }
  }

  async function load(seed) {
    if (state.running) {
      setStatus("実行中です。完了するまで再実行できません。", true);
      return false;
    }
    stop();
    state.historyId = "";
    updateAnalysisLink();
    const selectedProviderLabel = providerLabel(state.provider);
    setRunning(true, seed);
    let comparison;
    try {
      let sample = false;
      if (state.range === "single") {
        try {
          comparison = asComparison(await fetchJson(apiRunUrl("/api/ab-poc/run", seed, "A")));
        } catch (apiError) {
          if (state.provider !== "rule" || seed !== 1) throw apiError;
          comparison = asComparison(await fetchJson("sample-run-ab-seed-1.json"));
          sample = true;
        }
      } else {
        try {
          comparison = asComparison(await fetchJson(apiRunUrl("/api/ab-poc/compare", seed)));
        } catch (apiError) {
          if (state.provider !== "rule" || seed !== 1) throw apiError;
          comparison = asComparison(await fetchJson("sample-run-ab-seed-1.json"));
          sample = true;
        }
      }
      applyLoaded(comparison, { provider: state.provider, range: state.range, sample });
      const rangeLabel = resultRangeLabel(comparison, state.range);
      setStatus(sample
        ? `API未接続：固定サンプル ${rangeLabel} / seed ${comparison.root_seed}`
        : `完了：${selectedProviderLabel} ${rangeLabel} / seed ${comparison.root_seed}`);
    } catch (error) {
      setStatus(`読み込みエラー：${error.message}`, true);
    } finally {
      setRunning(false);
      const entries = await refreshHistory();
      if (comparison && state.provider !== "rule") {
        const saved = entries.find((entry) => (
          entry.provider === state.provider
          && String(entry.seed) === String(comparison.root_seed)
        ));
        if (saved) {
          state.historyId = String(saved.id);
          updateAnalysisLink();
        }
      }
    }
    return true;
  }

  document.querySelectorAll("[data-condition]").forEach((button) => button.addEventListener("click", () => {
    if (state.running) return;
    if ((state.loadedRange || state.range) === "single" && button.dataset.condition === "B") return;
    closeResult();
    stop(); state.condition = button.dataset.condition; state.step = 0; renderRunInfo(); render();
  }));
  els.provider.addEventListener("change", () => {
    if (state.running) return;
    closeResult({ restoreFocus: false });
    state.provider = els.provider.value;
    setStatus("次回実行の設定を変更しました。現在表示中の結果はそのままです。再実行で反映します");
  });
  els.range.addEventListener("change", () => {
    if (state.running) return;
    closeResult({ restoreFocus: false });
    state.range = els.range.value;
    state.condition = "A";
    renderRunInfo();
    render();
    setStatus("次回実行の設定を変更しました。現在表示中の結果はそのままです。再実行で反映します");
  });
  els.loadHistory.addEventListener("click", async () => {
    if (state.running) return;
    const historyId = els.history.value;
    if (!historyId) return;
    els.loadHistory.disabled = true;
    try {
      await loadHistoryEntry(historyId);
    } catch (error) {
      setStatus(`履歴の読み込みエラー：${error.message}`, true);
    } finally {
      els.loadHistory.disabled = false;
    }
  });
  els.history.addEventListener("change", () => {
    els.loadHistory.disabled = !els.history.value;
  });
  els.seedForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.running) {
      setStatus("実行中です。完了するまで再実行できません。", true);
      return;
    }
    const rawSeed = els.seed.value.trim();
    if (!/^[+-]?\d+$/.test(rawSeed)) { setStatus("seedは符号付き10進整数で入力してください。", true); return; }
    const seed = Number(rawSeed);
    if (!Number.isSafeInteger(seed)) { setStatus("seedは安全な整数の範囲で入力してください。", true); return; }
    void load(seed).catch((error) => {
      // Keep an unexpected exception from leaving the form locked forever.
      setRunning(false);
      setStatus(`予期しない実行エラー：${error.message}`, true);
    });
  });
  els.prev.addEventListener("click", () => { closeResult({ restoreFocus: false }); stop(); state.step -= 1; render(); });
  els.next.addEventListener("click", () => { closeResult({ restoreFocus: false }); stop(); state.step += 1; render(); });
  els.play.addEventListener("click", togglePlay);
  els.resultButton.addEventListener("click", openResult);
  els.resultCloseButton.addEventListener("click", () => closeResult());
  els.resultContinueButton.addEventListener("click", () => closeResult());
  els.resultReplayButton.addEventListener("click", () => {
    closeResult({ restoreFocus: false });
    stop();
    state.step = 0;
    render();
    togglePlay();
  });
  els.resultOverlay.addEventListener("click", (event) => {
    if (event.target === els.resultOverlay) closeResult();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && els.resultOverlay && !els.resultOverlay.hidden) {
      event.preventDefault();
      closeResult();
      return;
    }
    if (event.target.matches("input, button, select, textarea")) return;
    if (event.key === "ArrowLeft" && !els.prev.disabled) els.prev.click();
    if (event.key === "ArrowRight" && !els.next.disabled) els.next.click();
  });

  // Initial display is deliberately offline: use the newest saved LLM run,
  // or the bundled sample when no history exists. Only form submission calls
  // the simulation/API endpoint.
  loadInitial();
})();
