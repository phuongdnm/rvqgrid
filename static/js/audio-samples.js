// Audio sample comparison for speech separation.
//
// On a deployed/served page this reads static/audio/manifest.json (the source
// of truth produced alongside the audio files). When the page is opened straight
// from disk (file://), browsers block that fetch, so we transparently fall back
// to the same naming convention encoded below.

(function () {
  const MODELS = [
    { id: "codecformer", label: "Codecformer", ours: false },
    { id: "rvqgrid_dac", label: "RVQ-Grid (DAC)", ours: true },
    { id: "rvqgrid_encodec", label: "RVQ-Grid (EnCodec)", ours: true },
  ];
  const NUM_ITEMS_FALLBACK = 5;

  const labelFor = (id) =>
    (MODELS.find((m) => m.id === id) || {}).label ||
    id.replace(/_/g, " ");
  const isOurs = (id) => !!(MODELS.find((m) => m.id === id) || {}).ours;

  // Build the same item shape the manifest uses, from the naming convention.
  function generateItems() {
    const items = [];
    for (let i = 0; i < NUM_ITEMS_FALLBACK; i++) {
      const models = {};
      MODELS.forEach((m) => {
        models[m.id] = {
          source1: `audio/${m.id}/item${i}_source1hat.wav`,
          source2: `audio/${m.id}/item${i}_source2hat.wav`,
        };
      });
      items.push({
        id: `item${i}`,
        mix: `audio/mixture/item${i}_mix.wav`,
        groundTruth: {
          source1: `audio/mixture/item${i}_source1.wav`,
          source2: `audio/mixture/item${i}_source2.wav`,
        },
        models,
      });
    }
    return items;
  }

  async function loadItems() {
    try {
      const res = await fetch("static/audio/manifest.json", { cache: "no-cache" });
      if (res.ok) {
        const data = await res.json();
        if (data && Array.isArray(data.items) && data.items.length) {
          return data.items;
        }
      }
    } catch (e) {
      /* file:// or offline — use the fallback below */
    }
    return generateItems();
  }

  function audioCell(src) {
    return `<audio controls preload="none" src="static/${src}"></audio>`;
  }

  function renderComparison(item) {
    const rows = [
      {
        label: "Ground Truth",
        cls: "is-truth",
        s1: item.groundTruth.source1,
        s2: item.groundTruth.source2,
      },
    ];
    Object.keys(item.models).forEach((id) => {
      rows.push({
        label: labelFor(id),
        cls: isOurs(id) ? "is-ours" : "",
        s1: item.models[id].source1,
        s2: item.models[id].source2,
      });
    });

    const body = rows
      .map(
        (r) => `
        <tr class="${r.cls}">
          <th scope="row">${r.label}</th>
          <td>${audioCell(r.s1)}</td>
          <td>${audioCell(r.s2)}</td>
        </tr>`
      )
      .join("");

    return `
      <div class="audio-mixture">
        <span class="audio-mixture-label">Input Mixture</span>
        ${audioCell(item.mix)}
      </div>
      <div class="audio-table-wrap">
        <table class="audio-table">
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col">Speaker 1</th>
              <th scope="col">Speaker 2</th>
            </tr>
          </thead>
          <tbody>${body}</tbody>
        </table>
      </div>`;
  }

  function init(items) {
    const tabsEl = document.getElementById("sample-tabs");
    const comparisonEl = document.getElementById("audio-comparison");
    if (!tabsEl || !comparisonEl) return;

    let active = 0;
    const select = (i) => {
      active = i;
      Array.from(tabsEl.children).forEach((btn, idx) =>
        btn.classList.toggle("is-active", idx === i)
      );
      comparisonEl.innerHTML = renderComparison(items[i]);
    };

    tabsEl.innerHTML = "";
    items.forEach((_, i) => {
      const btn = document.createElement("button");
      btn.className = "sample-tab";
      btn.type = "button";
      btn.textContent = `Sample ${i + 1}`;
      btn.addEventListener("click", () => select(i));
      tabsEl.appendChild(btn);
    });

    select(0);
  }

  document.addEventListener("DOMContentLoaded", function () {
    loadItems().then(init);
  });
})();
