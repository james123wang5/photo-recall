"use strict";
// The server owns query/ranking revisions; UI selections always carry asset IDs.
const $ = (selector) => document.querySelector(selector);
const labels = {
  supported: "逐项较符合",
  conflict: "发现条件冲突",
  unknown: "无法判断",
  unreviewed: "尚未核对",
  error: "处理失败",
  excluded: "已排除",
};
let state,
  library,
  current,
  displayed,
  token,
  generation = 0,
  timer,
  jobTimer,
  offset = 0,
  activePhoto,
  crop,
  drag,
  submitting = false;
const PAGE_SIZE = 60;
function node(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => ($("#toast").hidden = true), 6000);
}
async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-PhotoRecall-Token": token,
          },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok)
    throw Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  return data;
}
function running(run = current) {
  return !!run && ["queued", "running", "pausing"].includes(run.status);
}
function remember() {
  if (library && current)
    localStorage.setItem("photorecall-flow-" + library.id, current.id);
}
function poll() {
  clearTimeout(timer);
  const stamp = generation;
  if (!current) return;
  timer = setTimeout(async () => {
    try {
      const next = await api("/api/flow/runs/" + current.id);
      if (stamp !== generation || next.revision < (current?.revision ?? 0))
        return;
      current = next;
      renderStatus();
      if (!displayed || !displayed.results.length) {
        displayed = next;
        offset = 0;
        renderResults();
      } else if (displayed.revision !== next.revision) {
        $("#apply").hidden = false;
      }
      if (running()) poll();
    } catch (e) {
      if (stamp === generation) toast(e.message);
    }
  }, 1800);
}
function renderStatus() {
  if (!current) return;
  $("#progress").hidden = false;
  $("#stage").textContent = current.stage;
  const c = current.coverage;
  $("#coverage").textContent =
    `本轮范围 ${c.total} 张 · 已处理 ${c.processed} 张 · 较符合 ${c.supported} · 无法判断 ${c.unknown} · 失败 ${c.error} · 未处理 ${c.unreviewed} · 你已排除 ${c.excluded || 0}`;
  $("#stop-reason").textContent =
    current.stop_reason || "正在处理；可以先浏览已有候选。";
  $("#pause").disabled = !running();
  $("#continue").disabled = running();
  $("#retry").disabled = running() || !c.error;
  $("#recover").disabled = !current.session.conditions.length;
  $("#warnings").textContent = current.warnings.join("；");
  $("#conditions").replaceChildren(
    ...current.session.conditions.map((c) => {
      const b = node("button", (c.soft ? "可能 · " : "") + c.label + " ×");
      b.title = "撤回这个条件";
      b.disabled = current.stage === "整理可修改线索" && running();
      b.onclick = () => submit({ remove_id: c.id });
      return b;
    }),
  );
  $("#reference").textContent = current.session.reference
    ? "已使用参考照片" +
      (current.session.reference.box ? "的框选局部" : "") +
      "；原有线索继续保留。"
    : "";
}
function renderResults() {
  if (!displayed) return;
  $("#gallery").replaceChildren(
    ...displayed.results.map((photo) => {
      const card = node("article", undefined, "card");
      card.dataset.state = photo.state;
      const button = node("button", undefined, "image-button");
      const img = node("img");
      img.src = photo.thumbnail;
      img.alt = photo.name;
      img.loading = "lazy";
      button.append(img);
      button.onclick = () => openPhoto(photo);
      const info = node("div", undefined, "card-info");
      info.append(
        node("span", labels[photo.state] || "候选", "badge"),
        node("div", photo.name, "card-name"),
        node(
          "small",
          photo.sources
            .map((s) => s.channel)
            .slice(0, 2)
            .join(" · "),
          "muted",
        ),
      );
      card.append(button, info);
      return card;
    }),
  );
  $("#result-count").textContent =
    `${displayed.result_total} 张可浏览 · 首轮候选池 ${displayed.candidate_count} 张`;
  $("#empty").hidden = !!displayed.results.length;
  $("#empty").textContent = running()
    ? "正在整理候选，请稍候。"
    : "当前视图没有照片；可以切回所有相关候选、修改线索或继续找。";
  $("#page").textContent =
    `${Math.floor(offset / PAGE_SIZE) + 1} / ${Math.max(1, Math.ceil(displayed.result_total / PAGE_SIZE))}`;
  $("#previous").disabled = offset === 0;
  $("#next").disabled = offset + PAGE_SIZE >= displayed.result_total;
  $("#apply").hidden = true;
}
async function loadPage(newOffset = 0) {
  if (!current) return;
  const stamp = generation;
  try {
    const run = await api(
      `/api/flow/runs/${current.id}?offset=${newOffset}&limit=${PAGE_SIZE}&filter_by=${encodeURIComponent($("#result-filter").value)}`,
    );
    if (stamp !== generation) return;
    current = run;
    displayed = run;
    offset = newOffset;
    renderStatus();
    renderResults();
  } catch (e) {
    toast(e.message);
  }
}
async function submit(extra = {}) {
  if (submitting) return;
  if (!library) {
    toast("先添加或选择照片库");
    return;
  }
  const message = extra.message ?? $("#query").value.trim();
  if (!message && !current && !extra.reference && !extra.action) return;
  const stamp = ++generation;
  submitting = true;
  clearTimeout(timer);
  $("#send").disabled = true;
  try {
    const body = {
      library_id: library.id,
      session_id: current?.session.id,
      expected_revision: current?.session.revision || 0,
      mode: $("#task-mode").value,
      message,
      ...extra,
    };
    const run = await api("/api/flow/search", body);
    if (stamp !== generation) return;
    current = displayed = run;
    offset = 0;
    $("#query").value = "";
    $("#result-filter").value = "all";
    renderStatus();
    renderResults();
    remember();
    poll();
  } catch (e) {
    if (stamp === generation) toast(e.message);
  } finally {
    submitting = false;
    $("#send").disabled = false;
  }
}
async function chooseLibrary(id) {
  const stamp = ++generation;
  clearTimeout(timer);
  clearTimeout(jobTimer);
  library = state.libraries.find((l) => l.id === id);
  current = displayed = null;
  offset = 0;
  $("#progress").hidden = true;
  $("#gallery").replaceChildren();
  $("#conditions").replaceChildren();
  $("#warnings").textContent = "";
  $("#reference").textContent = "";
  $("#library-name").textContent = library.name;
  document
    .querySelectorAll("#libraries button")
    .forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  localStorage.setItem("photorecall-flow-library", id);
  try {
    const recent = await api(`/api/flow/libraries/${id}/sessions`);
    if (stamp !== generation) return;
    const saved = localStorage.getItem("photorecall-flow-" + id);
    const found =
      recent.sessions.find((s) => s.current_run === saved) ||
      recent.sessions[0];
    if (found) {
      const run = await api("/api/flow/runs/" + found.current_run);
      if (stamp !== generation) return;
      current = displayed = run;
      $("#task-mode").value = run.session.mode;
      renderStatus();
      renderResults();
      if (running()) poll();
    } else {
      await submit({ action: "reset", message: "" });
    }
    watchIndex();
  } catch (e) {
    toast(e.message);
  }
}
function renderLibraries() {
  const libs = state.libraries.filter((l) => !l.is_demo);
  $("#libraries").replaceChildren(
    ...libs.map((lib) => {
      const b = node("button", `${lib.name} · ${lib.count} 张`);
      b.dataset.id = lib.id;
      b.onclick = () => chooseLibrary(lib.id);
      return b;
    }),
  );
}
async function watchIndex() {
  clearTimeout(jobTimer);
  if (!library) return;
  const id = library.id;
  try {
    const job = await api(`/api/libraries/${id}/job`);
    if (library?.id !== id) return;
    $("#job").hidden = job.status === "idle";
    if (job.status !== "idle") {
      const text = node(
        "p",
        `${job.current || "文件处理"} · ${job.done || 0}/${job.total || 0} · ${job.status}${job.error_count ? " · 失败 " + job.error_count + " 项" : ""}`,
      );
      $("#job").replaceChildren(text);
      if (["running", "scanning"].includes(job.status)) {
        const stop = node("button", "暂停建库");
        stop.onclick = async () => {
          try {
            await api(`/api/libraries/${id}/pause`, {});
            watchIndex();
          } catch (e) {
            toast(e.message);
          }
        };
        $("#job").append(stop);
        jobTimer = setTimeout(watchIndex, 1800);
      } else if (job.errors?.length) {
        $("#job").append(
          node(
            "p",
            job.errors
              .slice(-2)
              .map((e) => e.message)
              .join("；"),
            "muted",
          ),
        );
      }
    }
  } catch (e) {
    toast(e.message);
  }
}
function openPhoto(photo) {
  activePhoto = { ...photo, run_id: displayed.id };
  crop = drag = null;
  $("#crop-box").hidden = true;
  $("#photo-title").textContent = photo.name;
  $("#photo-image").src = "/api/assets/" + photo.id + "/original";
  $("#original").href = "/api/assets/" + photo.id + "/original";
  const ev = photo.evidence;
  $("#photo-evidence").textContent = ev
    ? ev.tool_error ||
      [
        ev.joint_evidence,
        ...(ev.conditions || []).map(
          (c) =>
            (current.session.conditions.find((x) => x.id === c.condition_id)
              ?.label || c.condition_id) +
            "：" +
            (labels[c.status] || c.status) +
            " · " +
            c.evidence +
            (c.entities ? "（" + c.entities + "）" : ""),
        ),
      ].join("\n")
    : "这张照片尚未逐项核对。";
  $("#inspect-photo").disabled = running();
  $("#reference-crop").disabled = true;
  $("#photo-dialog").showModal();
}
function selectedPhoto() {
  if (!activePhoto || activePhoto.run_id !== current?.id)
    throw Error("查询已改变，请重新打开当前结果中的照片");
  return activePhoto;
}
$("#photo-wrap").addEventListener("pointerdown", (event) => {
  if (!$("#photo-image").complete || !$("#photo-image").naturalWidth) return;
  const rect = $("#photo-image").getBoundingClientRect();
  drag = [
    Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  ];
  $("#photo-wrap").setPointerCapture(event.pointerId);
});
$("#photo-wrap").addEventListener("pointermove", (event) => {
  if (!drag) return;
  const rect = $("#photo-image").getBoundingClientRect();
  const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
  crop = [
    Math.min(x, drag[0]),
    Math.min(y, drag[1]),
    Math.max(x, drag[0]),
    Math.max(y, drag[1]),
  ];
  const b = $("#crop-box");
  b.hidden = false;
  b.style.left = crop[0] * 100 + "%";
  b.style.top = crop[1] * 100 + "%";
  b.style.width = (crop[2] - crop[0]) * 100 + "%";
  b.style.height = (crop[3] - crop[1]) * 100 + "%";
});
$("#photo-wrap").addEventListener("pointerup", () => {
  drag = null;
  $("#reference-crop").disabled =
    !crop || crop[2] - crop[0] < 0.01 || crop[3] - crop[1] < 0.01;
});
$("#photo-wrap").addEventListener("pointercancel", () => {
  drag = null;
});
$("#reference-photo").onclick = async () => {
  try {
    const p = selectedPhoto();
    $("#photo-dialog").close();
    await submit({ message: "", reference: { asset_id: p.id } });
  } catch (e) {
    toast(e.message);
  }
};
$("#reference-crop").onclick = async () => {
  try {
    const p = selectedPhoto();
    if (!crop) throw Error("先框选一个局部");
    $("#photo-dialog").close();
    await submit({ message: "", reference: { asset_id: p.id, box: crop } });
  } catch (e) {
    toast(e.message);
  }
};
$("#exclude-photo").onclick = async () => {
  try {
    const p = selectedPhoto();
    $("#photo-dialog").close();
    await submit({ message: "", exclude_id: p.id });
  } catch (e) {
    toast(e.message);
  }
};
$("#inspect-photo").onclick = async () => {
  try {
    const p = selectedPhoto();
    current = await api(`/api/flow/runs/${current.id}/continue`, {
      asset_ids: [p.id],
      regions: crop ? { [p.id]: [crop] } : null,
    });
    $("#photo-dialog").close();
    renderStatus();
    poll();
  } catch (e) {
    toast(e.message);
  }
};
$("#search-form").onsubmit = (event) => {
  event.preventDefault();
  if (!$("#query").value.trim()) return;
  const message = $("#query").value.trim();
  submit(
    /^(?:都不对|继续找|没找到|这批都不对)[。！!]?$/u.test(message)
      ? { action: "recover", message: "" }
      : {},
  );
};
$("#query").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#search-form").requestSubmit();
  }
});
$("#new-search").onclick = () => submit({ action: "reset", message: "" });
$("#recover").onclick = () => submit({ action: "recover", message: "" });
$("#pause").onclick = async () => {
  try {
    current = await api(`/api/flow/runs/${current.id}/pause`, {});
    renderStatus();
    poll();
  } catch (e) {
    toast(e.message);
  }
};
async function resume(retry = false) {
  try {
    current = await api(`/api/flow/runs/${current.id}/continue`, {
      retry_failed: retry,
    });
    renderStatus();
    poll();
  } catch (e) {
    toast(e.message);
  }
}
$("#continue").onclick = () => resume();
$("#retry").onclick = () => resume(true);
$("#apply").onclick = () => loadPage(0);
$("#previous").onclick = () => loadPage(Math.max(0, offset - PAGE_SIZE));
$("#next").onclick = () => loadPage(offset + PAGE_SIZE);
$("#result-filter").onchange = () => loadPage(0);
document
  .querySelectorAll("[data-close]")
  .forEach((b) => (b.onclick = () => b.closest("dialog").close()));
$("#add-library").onclick = async () => {
  try {
    const picked = await api("/api/pick-folder", {});
    if (picked.cancelled || !picked.path) return;
    const lib = await api("/api/libraries", { path: picked.path });
    state = await api("/api/state");
    renderLibraries();
    await chooseLibrary(lib.id);
    toast("文件夹已添加，点击更新文件建立文件清单。");
  } catch (e) {
    toast(e.message);
  }
};
$("#scan").onclick = async () => {
  if (!library) return;
  try {
    await api(`/api/flow/libraries/${library.id}/scan`, {});
    watchIndex();
  } catch (e) {
    toast(e.message);
  }
};
$("#index-open").onclick = async () => {
  if (!library) return;
  try {
    const coverage = await api(`/api/flow/libraries/${library.id}/coverage`);
    $("#index-coverage").textContent = JSON.stringify(coverage, null, 2);
    $("#index-dialog").showModal();
  } catch (e) {
    toast(e.message);
  }
};
$("#index-start").onclick = async () => {
  if (!library) return;
  try {
    const components = [
      ...document.querySelectorAll("[name=component]:checked"),
    ].map((n) => n.value);
    await api(`/api/flow/libraries/${library.id}/index`, { components });
    $("#index-dialog").close();
    watchIndex();
  } catch (e) {
    toast(e.message);
  }
};
$("#settings-open").onclick = async () => {
  try {
    const data = await api("/api/flow/config");
    const c = data.config;
    $("#primary-model").value = c.primary;
    $("#reranker-model").value = c.reranker;
    $("#vision-model").value = c.vision_model;
    for (const [id, key] of [
      ["planner-enabled", "planner"],
      ["auto-verify", "auto_verify"],
      ["auto-rerank", "auto_rerank"],
      ["grounding-enabled", "grounding_enabled"],
      ["people-enabled", "people_enabled"],
    ])
      $("#" + id).checked = c[key];
    $("#verify-budget").value = c.verify_budget;
    $("#time-slice").value = c.time_slice;
    $("#model-status").textContent = JSON.stringify(data.models, null, 2);
    $("#settings-dialog").showModal();
  } catch (e) {
    toast(e.message);
  }
};
$("#settings-form").onsubmit = async (event) => {
  event.preventDefault();
  try {
    await api("/api/flow/config", {
      primary: $("#primary-model").value,
      reranker: $("#reranker-model").value,
      vision_model: $("#vision-model").value.trim(),
      planner: $("#planner-enabled").checked,
      auto_verify: $("#auto-verify").checked,
      auto_rerank: $("#auto-rerank").checked,
      grounding_enabled: $("#grounding-enabled").checked,
      people_enabled: $("#people-enabled").checked,
      verify_budget: Number($("#verify-budget").value),
      time_slice: Number($("#time-slice").value),
    });
    toast("已保存，用于下一轮新搜索。");
  } catch (e) {
    toast(e.message);
  }
};
$("#people-open").onclick = async () => {
  if (!library) return;
  try {
    const data = await api(`/api/flow/libraries/${library.id}/people`);
    $("#people-list").replaceChildren(
      ...data.people.map((person) => {
        const row = node("div");
        const img = node("img");
        img.src = `/api/assets/${person.example}/thumbnail`;
        const name = node("input");
        name.value = person.name;
        name.placeholder = "为这个分组命名";
        const save = node("button", "保存名称");
        save.onclick = async () => {
          try {
            await api(`/api/flow/people/${person.id}/name`, {
              name: name.value,
            });
            toast("名称已保存，相似聚类仍可能有误。");
          } catch (e) {
            toast(e.message);
          }
        };
        const use = node("button", `找此人 · ${person.count} 张`);
        use.onclick = () => {
          $("#settings-dialog").close();
          submit({ message: "", person_id: person.id });
        };
        row.append(img, name, save, use);
        return row;
      }),
    );
  } catch (e) {
    toast(e.message);
  }
};
$("#history").addEventListener("toggle", async () => {
  if (!$("#history").open || !current) return;
  try {
    const data = await api(`/api/flow/runs/${current.id}/actions`);
    $("#actions").replaceChildren(
      ...data.actions.map((a) =>
        node(
          "p",
          (a.reason ||
            {
              created: "开始搜索",
              recall: "合并候选",
              inspect: "核对照片",
              stopped: "本轮停止",
            }[a.kind] ||
            a.kind) +
            (a.outcome ? " · " + (labels[a.outcome] || a.outcome) : "") +
            (a.asset_id ? " · " + a.asset_id : ""),
        ),
      ),
    );
  } catch (e) {
    toast(e.message);
  }
});
$("#export-run").onclick = async () => {
  if (!current) return;
  try {
    const data = await api(`/api/flow/runs/${current.id}/actions`);
    const blob = new Blob(
      [JSON.stringify({ run: current, actions: data.actions }, null, 2)],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = node("a");
    a.href = url;
    a.download = `photorecall-${current.id}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) {
    toast(e.message);
  }
};
(async () => {
  try {
    state = await api("/api/state");
    token = state.token;
    renderLibraries();
    const available = state.libraries.filter((l) => !l.is_demo);
    const saved = localStorage.getItem("photorecall-flow-library");
    const selected = available.find((l) => l.id === saved) || available[0];
    if (selected) await chooseLibrary(selected.id);
  } catch (e) {
    toast(e.message);
  }
})();
